import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer_prompts as vit
from utils.functions import assign_gpu
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils, dicmor
from clip import clip
import json
import random
class TopoPromptLearner(nn.Module):
    def __init__(self, classnames, prompt_topo): # prompt_topo：各类的拓扑结构
        super().__init__()

        self.classnames = classnames #类名列表
        self.dtype = torch.float32 # CLIP模型的数据类型
        self.n_set = 5 # number of descriptions for each category 每个类别的描述数量
        self.n_tpro = 2 # prompt length 提示长度
        self.layers = 6 # CLIP模型中的Transformer层数
        
        # layer-wise scalar to weight indicating the strength of the relationship of entity-entity pairs and entity-attribute pairs
        # 用于权衡实体-实体和实体-属性关系的层级标量 , 每个层都有一个单独的标量参数
        self.e2e_scal = nn.Parameter(torch.zeros(self.layers, 1, 1, 1)) 
        self.e2a_scal = nn.Parameter(torch.zeros(self.layers, 1, 1, 1))
        
        # 用于存储每个类别的实体-实体和实体-属性关系的注意力矩阵 在这个字典中，每个类名（'cat', 'dog', 'bird'）都对应一个空的列表
        self.attns_e2e = {classname: [] for classname in classnames}
        self.attns_e2a = {classname: [] for classname in classnames}

        # 用于生成提示文本的前缀
        prompt_prefix = " ".join(["X"] * (self.n_tpro + self.n_set))

        for classname in classnames:
            topos = prompt_topo[classname]
            for id in range(self.n_set):
                # generate text with classname, entities and attributes
                txt = self.generate_text(classname, prompt_prefix, topos[id])
                tokens = clip.tokenize(txt, truncate=True)[0]
                
                # generate pair-wise relationships
                e2e, e2a = self.extract_relationships(tokens, topos[id])

                # create attention matrix based on pair-wise relationships
                attn_e2e = self.create_attention_matrix(tokens, e2e)
                attn_e2a = self.create_attention_matrix(tokens, e2a)

                # save attention matrices
                self.attns_e2e[classname].append(attn_e2e)
                self.attns_e2a[classname].append(attn_e2a)

    # generate text with classname, entities and attributes
    # 根据类名、提示前缀和拓扑结构生成包含实体和属性的文本 "X X X X X X X X class_name. entity1, entity2, entity3. attribute1, attribute2."
    def generate_text(self, classname, prompt_prefix, topo):
        entities = [w.lower() for w in topo['Entities']]
        attributes = [w.lower() for w in topo['Attributes']]
        txt = prompt_prefix + " " + classname + ". " + ", ".join(entities) + ". " + ", ".join(attributes) + "."
        return txt

    # generate pair-wise relationships from topological structure
    # 从拓扑结构中提取实体-实体和实体-属性关系，并对其进行标记对齐
    def extract_relationships(self, tokens, topo):
        entities = [w.lower() for w in topo['Entities']]
        attributes = [w.lower() for w in topo['Attributes']]
        e2e, e2a = [], []

        for w in topo['Entity-to-Entity Relationships']:
            if w['entity1'].lower() in entities and w['entity2'].lower() in entities:
                e1 = list(self.align(tokens, self.truncate(clip.tokenize(w['entity1']))[0]))  # 这个关系中，entity1的若干token，align 函数找到这些 token 在 tokens 序列中的位置 假设是 [7, 8]
                e2 = list(self.align(tokens, self.truncate(clip.tokenize(w['entity2']))[0]))  # 这个关系中，entity2的若干token，align 函数找到这些 token 在 tokens 序列中的位置 假设是 [9, 10, 11]
                e2e.append([e1, e2]) # e2e就添加一条[[7, 8], [9, 10, 11]]

        for w in topo['Entity-to-Attribute Relationships']:
            if w['entity'].lower() in entities and w['attribute'].lower() in attributes:
                e1 = list(self.align(tokens, self.truncate(clip.tokenize(w['entity']))[0]))
                e2 = list(self.align(tokens, self.truncate(clip.tokenize(w['attribute']))[0]))
                e2a.append([e1, e2])
        return e2e, e2a

    # create attention matrix based on pair-wise relationships
    # 基于成对关系创建注意力矩阵
    def create_attention_matrix(self, tokens, relationships):
        n_tokens = len(tokens)
        attn = torch.zeros(n_tokens, n_tokens).cuda()

        for e in relationships:
            d11 = torch.tensor([[i] for i in e[0]]).type(torch.long)
            d21 = torch.tensor([e[1] for _ in range(len(e[0]))]).type(torch.long)
            d12 = torch.tensor([[i] for i in e[1]]).type(torch.long)
            d22 = torch.tensor([e[0] for _ in range(len(e[1]))]).type(torch.long)
            
            attn[d11, d21] += 1
            attn[d12, d22] += 1
        return attn

    # truncate token sequence according to EOS token
    # 根据结束标记截断标记序列
    def truncate(self, array):
        return array[:, 1:torch.argmax(array)]

    # find a sequence that matches the target token(s)
    # 找到与目标标记匹配的序列
    def align(self, seq1, seq2):
        for idx in range(len(seq1) - len(seq2) + 1):
            if seq1[idx:idx + len(seq2)].equal(seq2):
                return range(idx, idx + len(seq2))
        return []

    def forward(self): # 对于每个类别，使用两个可训练标量权重生成的注意力矩阵，并返回这些矩阵
        attns = {}
        for classname in self.classnames:
            classname = classname.replace("_", " ")
            # weight generated matrices with two learnable scalars
            attns[classname] = self.e2e_scal * torch.stack(self.attns_e2e[classname]).cuda() + \
                               self.e2a_scal * torch.stack(self.attns_e2a[classname]).cuda()
        return attns

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        dicmor_args= {'model_name': 'dicmor', 'dataset_name': 'mosi', 'featurePath': 'dataset/MOSI/aligned_50.pkl', 'feature_dims': [512,217], 'train_samples': 1284, 'num_classes': 16, 'language': 'en', 'KeyEval': 'Loss', 'need_data_aligned': True, 'need_model_aligned': True, 'early_stop': 10, 'use_bert': False, 'use_finetune': True, 'attn_mask': True, 'update_epochs': 8, 'attn_dropout_a': 0.2, 'attn_dropout_v': 0.0, 'relu_dropout': 0.0, 'embed_dropout': 0.2, 'res_dropout': 0.0, 'dst_feature_dim_nheads': [32, 8], 'batch_size': config["batch_size"], 'learning_rate': 0.0001, 'nlevels': 4, 'conv1d_kernel_size_l': 5, 'conv1d_kernel_size_a': 5, 'conv1d_kernel_size_v': 5, 'text_dropout': 0.5, 'attn_dropout': 0.3, 'output_dropout': 0.5, 'grad_clip': 0.6, 'patience': 5, 'weight_decay': 0.005, 'transformers': 'bert', 'pretrained': 'bert-base-uncased', 'mode': 'train', 'mr': 0.1, 'model_save_path': 'pt/dicmor-mosi.pth', 'device': assign_gpu([0]), 'train_mode': 'classification', 'feature_T': '', 'feature_A': '', 'feature_V': '', 'cur_seed': 1}
        #self.dicmor= getattr(dicmor, 'DICMOR')(dicmor_args)
        self.classnames = [
    "apple pie",
    "baby back ribs",
    "baklava",
    "beef carpaccio",
    "beef tartare",
    "beet salad",
    "beignets",
    "bibimbap",
    "bread pudding",
    "breakfast burrito",
    "bruschetta",
    "caesar salad",
    "cannoli",
    "caprese salad",
    "carrot cake",
    "ceviche",
    "cheese plate",
    "cheesecake",
    "chicken curry",
    "chicken quesadilla",
    "chicken wings",
    "chocolate cake",
    "chocolate mousse",
    "churros",
    "clam chowder",
    "club sandwich",
    "crab cakes",
    "creme brulee",
    "croque madame",
    "cup cakes",
    "deviled eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs benedict",
    "escargots",
    "falafel",
    "filet mignon",
    "fish and chips",
    "foie gras",
    "french fries",
    "french onion soup",
    "french toast",
    "fried calamari",
    "fried rice",
    "frozen yogurt",
    "garlic bread",
    "gnocchi",
    "greek salad",
    "grilled cheese sandwich",
    "grilled salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot and sour soup",
    "hot dog",
    "huevos rancheros",
    "hummus",
    "ice cream",
    "lasagna",
    "lobster bisque",
    "lobster roll sandwich",
    "macaroni and cheese",
    "macarons",
    "miso soup",
    "mussels",
    "nachos",
    "omelette",
    "onion rings",
    "oysters",
    "pad thai",
    "paella",
    "pancakes",
    "panna cotta",
    "peking duck",
    "pho",
    "pizza",
    "pork chop",
    "poutine",
    "prime rib",
    "pulled pork sandwich",
    "ramen",
    "ravioli",
    "red velvet cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed salad",
    "shrimp and grits",
    "spaghetti bolognese",
    "spaghetti carbonara",
    "spring rolls",
    "steak",
    "strawberry shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna tartare",
    "waffles"
]
        f_topo = '/data/lhy/HPT/data/gpt_data/structure/Food101.json'
        

        with open(f_topo, 'r') as f:
            text_topos = json.load(f)
        self.topo_prompt_learner = TopoPromptLearner(self.classnames, text_topos)
        
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
            and not self.hparams.config["finetune_first"]
        ):

            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # since the pre-trained max_text_len is 40,
            # we upsample the weight of position embedding to determined max_text_len
            if config["max_text_len"] != 40:
                state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,-1)
                pos_emb = state_dict['text_embeddings.position_embeddings.weight']
                pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(config["max_text_len"],768), mode='bilinear').squeeze()
                state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.hatememes_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.food101_classifier.apply(objectives.init_weights)               
            
        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.mmimdb_classifier.apply(objectives.init_weights)  
            
        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)            
            print("use pre-finetune model")
  
        self.prompt_type = self.hparams.config["prompt_type"]
        prompt_length = self.hparams.config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams.config["hidden_size"]
        self.learnt_p = self.hparams.config["learnt_p"]
        self.prompt_layers = self.hparams.config["prompt_layers"]
        self.multi_layer_prompt = self.hparams.config["multi_layer_prompt"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1
        from timm.models.layers import trunc_normal_
        '''
        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_prompt[:,0:1,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            complete_prompt[:,prompt_length//2:prompt_length//2+1,:].fill_(1)
        self.complete_prompt = nn.Parameter(complete_prompt)
        '''
        missing_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_text_prompt[:,2:3,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            missing_text_prompt[:,prompt_length//2+2:prompt_length//2+3,:].fill_(1)
        self.missing_text_prompt = nn.Parameter(missing_text_prompt)

        missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_img_prompt[:,1:2,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            missing_img_prompt[:,prompt_length//2+1:prompt_length//2+2,:].fill_(1)
        self.missing_img_prompt = nn.Parameter(missing_img_prompt)

        if not self.learnt_p:
            #self.complete_prompt.requires_grad=False
            self.missing_text_prompt.requires_grad=False           
            self.missing_img_prompt.requires_grad=False

        #print(self.complete_prompt)
        print(self.missing_img_prompt)
        print(self.missing_text_prompt)

        for param in self.transformer.parameters():
            param.requires_grad=False
        for param in self.text_embeddings.parameters():
            param.requires_grad=False
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=False

        # 确保 LoRA 的参数可训练
        for name, param in self.transformer.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        self.records = {}

    
    def calculate_ortho_loss(self, P_s, P_t, epsilon=1e-8):
        """
        计算正交损失 L_ortho
        """
        # 扁平化操作
        P_s_flat = P_s.flatten(start_dim=1)
        P_t_flat = P_t.flatten(start_dim=1)

        # 计算点积
        dot_product = torch.abs(torch.sum(P_s_flat * P_t_flat, dim=1))

        # 计算两个向量的L2范数的乘积
        norm_product = torch.norm(P_s_flat, p=2, dim=1) * torch.norm(P_t_flat, p=2, dim=1)

        # 避免除以0，确保分母不为零
        denominator = torch.clamp(norm_product, min=epsilon)

        # 计算L_ortho
        L_ortho = dot_product / denominator

        # L_ortho 是一个向量，其中包含了每个样本的损失，取平均得到最终的损失值
        return L_ortho.mean()
    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        is_train=None,
    ):
        attns = self.topo_prompt_learner()
        #print("self.topo_prompt_learner.e2e_scal=",self.topo_prompt_learner.e2e_scal)
        #print("self.topo_prompt_learner.e2a_scal=",self.topo_prompt_learner.e2a_scal)
        attn = []
        flag=True
        n_set=5
        if flag:
            for name in self.classnames:
                id = random.randint(0, n_set-1)
                attn.append(attns[name][:, id])

        else:
            for name in self.classnames:
                # We leverage all structures from descriptions as a part of input respectively during evaluation.
                for id in range(n_set):
                    attn.append(attns[name][:, id])
        
        attn = torch.stack(attn, dim=0)  # Shape of attn: torch.Size([101, 6, 77, 77])

        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        img = batch[imgkey][0]     
        

        if image_embeds is None and image_masks is None:
                   
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
            
            # deal with zero input images
#             for idx in range(len(img)):
#                 if len(torch.unique(img[idx])) <= 2:
#                     image_embeds[idx,1:].fill_(0)
#                     image_masks[idx,1:].fill_(0)
#                     image_embeds[idx,1:].fill_(1)

        else:
            patch_index, image_labels = (
                None,
                None,
            )
            
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        
        batch_size, current_length, feature_dim = image_embeds.shape
        target_length=217
        if current_length > target_length:
            pad_image_embeds = image_embeds[:, :target_length, :]  # 如果当前长度超出目标长度，进行裁剪
        elif current_length < target_length:
            # 如果当前长度小于目标长度，进行填充
            padding_size = target_length - current_length
            padding = torch.zeros(batch_size, padding_size, feature_dim, device=image_embeds.device, dtype=image_embeds.dtype)
            pad_image_embeds = torch.cat([image_embeds, padding], dim=1)  # 在第二维度上进行拼接
        else:
            pad_image_embeds = image_embeds
        #print("text_embeds.shape=",text_embeds.shape)#food[8, 512, 768]   mmimdb [8, 1024, 768]    hatefull[10, 128, 768]
        #print("image_embeds.shape=",image_embeds.shape)# food[8, 217, 768]  [16, 229, 768]  mmimdb[8, 229, 768] [8, 217, 768] [8, 205, 768])  hatefull[10, 217, 768]
        #out_put=self.dicmor(text_embeds,pad_image_embeds,self.hparams.config['mt_index'],batch["missing_type"])
        # instance wise missing aware prompts
        '''
        if self.hparams.config['mt_index']==1: #图像缺失
            rec_v=out_put['rec_v']
            #print("infer内部：rec_v=",rec_v.shape) #[16, 32, 764]
            # 线性投影层匹配 rec_v 到 img_tokens 的维度
            rec_v = rec_v.transpose(1, 2) # [16, 764, 32]
            projection1 = nn.Linear(rec_v.size(2), target_length).to(rec_v.device)
            rec_v = projection1(rec_v) # [16, 764, 217]
            rec_v = rec_v.transpose(1, 2) # [16, 217, 764]
            projection2 = nn.Linear(rec_v.size(2), 768).to(rec_v.device)
            rec_v = projection2(rec_v) # [16, 217, 768]
            #mean_rec_v = torch.mean(rec_v, dim=0)
            #print("最终rec_v=",rec_v.shape)
            # 交换img_tokens的前两维
            ###遍历single_missing_types，根据不同的缺失情况，更改对应位置的img_tokens和txt_tokens
            for i in range(len(img)):
                if batch["missing_type"][i] == 2:
                    #print("最终image_embeds=",image_embeds.shape)
                    #print("rec_v.size()=",rec_v.size())# ([128, 49, 512])
                    
                    if rec_v.size(1) < current_length:
                        image_embeds[i,:rec_v.size(1),:] = rec_v[i]   #rec_v[i] 不用mean好像会稍微高那么一点点
                    else:
                        image_embeds[i,:,:] = rec_v[i][:image_embeds.size(1),:]
                    #img_cls[i,:] = torch.mean(rec_v[i], dim=0) # 用序列平均作为img_cls
            # 恢复形状以供后续模型使用  
        elif self.hparams.config['mt_index']==0:  #文本缺失  food的文本目标维度是[8,512,768]
            rec_l=out_put['rec_l']
            #print("rec_l=",rec_l.shape) # food [8, 32, 764]
            rec_l = rec_l.transpose(1, 2) # [8, 764, 32]
            # 线性投影层匹配 rec_l 到 txt_tokens 的维度
            target_length_food=512
            target_length_mmimdb=1024
            target_length_hatefull=128
            projection1 = nn.Linear(rec_l.size(2), target_length_food).to(rec_l.device)
            rec_l = projection1(rec_l) #  [8, 764, 512]
            rec_l = rec_l.transpose(1, 2) # [8, 512, 764]
            projection2 = nn.Linear(rec_l.size(2), 768).to(rec_l.device)
            rec_l = projection2(rec_l) # [16, 217, 768]
    
            ###遍历single_missing_types，根据不同的缺失情况，更改对应位置的img_tokens和txt_tokens
            for i in range(len(img)):
                if batch["missing_type"][i] == 1:
                    text_embeds[i,:,:] = torch.mean(rec_l[i] + text_embeds[i,:,:],dim=0)
                    #txt_eos[i,:] = torch.mean(rec_l[i], dim=0) # 用序列平均作为txt_eos
            
            #print("text_embeds.shape=",text_embeds.shape)
            #print("txt_eos=",txt_eos.shape)
        elif self.hparams.config['mt_index']==2:
            rec_v=out_put['rec_v']
            #print("infer内部：rec_v=",rec_v.shape) #[16, 32, 764]
            # 线性投影层匹配 rec_v 到 img_tokens 的维度
            rec_v = rec_v.transpose(1, 2) # [16, 764, 32]
            projection1 = nn.Linear(rec_v.size(2), target_length).to(rec_v.device)
            rec_v = projection1(rec_v) # [16, 764, 217]
            rec_v = rec_v.transpose(1, 2) # [16, 217, 764]
            projection2 = nn.Linear(rec_v.size(2), 768).to(rec_v.device)
            rec_v = projection2(rec_v) # [16, 217, 768]

            rec_l=out_put['rec_l']
            #print("rec_l=",rec_l.shape) # food [8, 32, 764]
            rec_l = rec_l.transpose(1, 2) # [8, 764, 32]
            # 线性投影层匹配 rec_l 到 txt_tokens 的维度
            target_length_food=512
            target_length_mmimdb=1024
            target_length_hatefull=128

            projection1 = nn.Linear(rec_l.size(2), target_length_food).to(rec_l.device)
            rec_l = projection1(rec_l) #  [8, 764, 512]
            rec_l = rec_l.transpose(1, 2) # [8, 512, 764]
            projection2 = nn.Linear(rec_l.size(2), 768).to(rec_l.device)
            rec_l = projection2(rec_l) # [16, 217, 768]
            for i in range(len(img)):
                if batch["missing_type"][i] == 2:
                    if rec_v.size(1) < current_length:
                        image_embeds[i,:rec_v.size(1),:] = rec_v[i]   #rec_v[i] 不用mean好像会稍微高那么一点点
                    else:
                        image_embeds[i,:,:] = rec_v[i][:image_embeds.size(1),:]
                if batch["missing_type"][i] == 1:
                    text_embeds[i,:,:] = torch.mean(rec_l[i] + text_embeds[i,:,:],dim=0)
        '''
        prompts = None
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                #prompt = self.complete_prompt      
                prompt = self.missing_text_prompt +  self.missing_img_prompt     
            elif batch["missing_type"][idx] == 1:
                prompt = self.missing_text_prompt
            elif batch["missing_type"][idx] == 2:
                prompt = self.missing_img_prompt
                
            if prompt.size(0) != 1:
                prompt = prompt.unsqueeze(0)
            
            if prompts is None:
                prompts = prompt
            else:
                prompts = torch.cat([prompts, prompt], dim=0)
        
        if self.learnt_p:
            if self.prompt_type=='attention':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length//2, dtype=prompts.dtype, device=prompts.device).long()
            elif self.prompt_type=='input':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length*len(self.prompt_layers), dtype=prompts.dtype, device=prompts.device).long()
        else:
            prompt_masks = torch.ones(prompts.shape[0], self.prompt_length, dtype=prompts.dtype, device=prompts.device).long()   
        
        co_masks = torch.cat([prompt_masks, text_masks, image_masks], dim=1)
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        x = co_embeds.detach()
        loss_ortho = self.calculate_ortho_loss(self.missing_img_prompt, self.missing_text_prompt)
        for i, blk in enumerate(self.transformer.blocks):
            if i in self.prompt_layers:
                if self.multi_layer_prompt:
                    x, _attn = blk(x, mask=co_masks, 
                                   prompts=prompts[:,self.prompt_layers.index(i)], 
                                   learnt_p=self.learnt_p,
                                   prompt_type=self.prompt_type,
                                   attn_m=attn[:, i],
                                   label=batch["label"]
                                   )
                else:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        
        if self.prompt_type == 'input':
            total_prompt_len = len(self.prompt_layers)* prompts.shape[-2]
        elif self.prompt_type == 'attention':
            total_prompt_len = prompts.shape[-2]
        
        text_feats, image_feats = (
            x[:,total_prompt_len : total_prompt_len+text_embeds.shape[1]],
            x[:, total_prompt_len+text_embeds.shape[1] :],
        )
        if self.prompt_type == 'input':
            cls_feats = self.pooler(x[:,total_prompt_len:total_prompt_len+1])   
#         cls_feats = self.pooler(x[:,:prompts.size(1)].mean(dim=1,keepdim=True))
        elif self.prompt_type == 'attention':
            cls_feats = self.pooler(x)
            
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
#            "loss_rec":out_put["loss_rec"],
            "loss_ortho":loss_ortho,
#            "log_p_v_loss":out_put["log_p_v"],
 #           "log_p_l_loss":out_put["log_p_l"]
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))
            
        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))
            
        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))              

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        
        '''
        print("output[loss_rec]=",output["loss_rec"])
        
        print("output[mmimdb_loss]=",output["mmimdb_loss"])
        
        print("output[log_p_v_loss]=",output["log_p_v_loss"])
        print("output[log_p_l_loss]=",output["log_p_l_loss"])
        '''
        return total_loss
    """def on_after_backward(self):
        # 在这里检查梯度
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad norm = {param.grad.norm().item()}")"""
    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)
#         print('missing_img:', self.missing_img_prompt[0,0:3,0:8])
#         print('missing_text:', self.missing_text_prompt[0,0:3,0:8])
#         print('complete:', self.complete_prompt[0,0:3,0:8])

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
