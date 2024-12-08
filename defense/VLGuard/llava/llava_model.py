import os
import torch
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration,AutoConfig,AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import BatchFeature
from ..multimodalmodel import MultiModalModel
from PIL import Image
class LLaVA_v1_5(MultiModalModel):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LlavaForConditionalGeneration.from_pretrained("/home/yjj/models/llava-v1.5").to(self.device)
        self.processor = AutoProcessor.from_pretrained("/home/yjj/models/llava-v1.5")
        self.normalize = transforms.Normalize(mean=self.processor.image_processor.image_mean, std=self.processor.image_processor.image_std)
        self.preprocess = lambda x: self.processor.image_processor.preprocess(x, do_normalize=False, return_tensors='pt')['pixel_values'][0]
        self.tokenizer_kwargs = {"padding": False, "truncation": None, "max_length": None}
        self.config=AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.conv_mode='llava_v0'
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.name=os.path.basename(model_path)

    def generate(self, test_cases, image_dir, **generation_kwargs):
        generation_kwargs.setdefault('do_sample', False)
        generation_kwargs.setdefault('num_beams', 1)
        assert generation_kwargs['do_sample'] is False, "do_sample should be False"
        assert generation_kwargs['num_beams'] == 1, "num_beams should be 1"
        outputs = []
        for case in test_cases:
            image, prompt = case
            if isinstance(image, str):
                raw_image = Image.open(os.path.join(image_dir, image))
            # assert torch.max(image) <= 1.0 and torch.min(image) >= 0.0, "Image tensor values are not in the range [0, 1]"
            
            # assert image.size() == torch.Size([3, 336, 336]), "Image shape is not as expected"
            conv_prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
            # text_inputs = self.processor.tokenizer(conv_prompt, return_tensors='pt', **self.tokenizer_kwargs).to(self.device)
            # pixel_values = self.normalize(image).unsqueeze(0).to(self.device)
            # inputs = BatchFeature(data={**text_inputs, "pixel_values": pixel_values})
            inputs = self.processor(conv_prompt, raw_image, return_tensors='pt').to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=1)
            output = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            outputs.append(output.rsplit('ASSISTANT:', 1)[-1].strip())
            # generate_ids = self.model.generate(**inputs, **generation_kwargs)
            # output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # outputs.append(output.rsplit('ASSISTANT:', 1)[-1].strip())
        return outputs

    def query(self, image, prompt):
        input = self.processor(text=f"Human: {prompt} <image> Assistant: ", images=image, return_tensors="pt").to(self.device)
        output = self.model.generate(**input,max_length=300)
        outnpy = output.to("cpu").numpy()
        answer = self.processor.decode(outnpy[0], skip_special_tokens=True)
        answer = answer.split('Assistant: ')[-1].strip()
        return answer

    def get_response(self, qs, defense_query, full_image_path):
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,IMAGE_PLACEHOLDER
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        qs=qs+defense_query
        # if self.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, 32000, return_tensors='pt').unsqueeze(0).to(self.device)
        # print('input_ids',input_ids)
        image = Image.open(full_image_path)
        # print(image)
        image_tensor = self.preprocess(image)
        inputs = self.processor(prompt, image, return_tensors='pt').to(self.device)
        # inputs.input_ids[0][10]=
        # inputs=tokenizer_image_token(prompt,self.tokenizer,IMAGE_TOKEN_INDEX,return_tensors=True)
        # print(inputs)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # print(self.tokenizer.decode(inputs.input_ids[0]))
        with torch.inference_mode():
            output_ids = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True
                )
        # print('output_ids',output_ids)
        # print('output_ids',output_ids.shape[1])
        input_token_len = inputs.input_ids.shape[1]
        if output_ids.shape[1]<=input_token_len:
            # print('self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)',self.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
            return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            print('stop str')
            outputs = outputs[:-len(stop_str)]  
        return outputs
    
    def adashield_defense(self,attack_method):
        table_dir=os.path.join('/home/yjj/MLLM-Jailbreak-evaluation-MMJ-Bench/AdaShield/final_tables',self.name,f'{self.name}_{attack_method}.csv')
        # defense=csv.DictReader(table)
        # next(defense)
        # defense_prompt={}
        # for item in defense:
        #     defense_prompt[item['BehaviorID']]=item['defense_prompt_list']
                    
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        data =open(os.path.join('/home/yjj/MLLM-Jailbreak-evaluation-MMJ-Bench/data/behavior_datasets/harmbench_behaviors_text_all.csv'),'r')
        data_dict=csv.DictReader(data)
        next(data_dict)
        data_sample={}
        
        test_cases_path=os.path.join('/home/yjj/MLLM-Jailbreak-evaluation-MMJ-Bench/test_cases',attack_method,f'{attack_method}_test_cases.json')
        # print(test_cases_path)
        with open(test_cases_path,'r') as test_cases_file:
            test_cases=json.load(test_cases_file)
        # print(test_cases)
        defense_prompt_pool, embedding_pool, image_pool  = get_all_defense_example_pool(clip_model, clip_preprocess, table_dir, defense_number=-1,device=self.device)
        i=0
        for item in data_dict:
            
            if i>=199 :#or item['BehaviorID'] not in list(test_cases.keys()) :
                print('break!!!!!!!!!!!!!!!!!!!!!!!!!')
                break
            data_sample[item['BehaviorID']]=[]
            # data_sample[item['BehaviorID']]['behavior']=item['Behavior']
            full_image_path=os.path.join('/home/yjj/MLLM-Jailbreak-evaluation-MMJ-Bench/test_cases',attack_method,'images',f"{test_cases[item['BehaviorID']][0]}")
            query=test_cases[item['BehaviorID']][1]
            sample_embedding =get_clip_embedding(clip_model, clip_preprocess,full_image_path,query ,device=self.device) #(args, conv_mode, tokenizer, model, image_processor, full_image_path, query)
            defense_query, best_simiarity , best_similarity = retrival_defense_prompt(defense_prompt_pool,image_pool, retrival_type= "sample-wise", sample_embedding=sample_embedding, embedding_pool=embedding_pool)
            image = Image.open(full_image_path).convert('RGB')
            text_prompt = query + defense_query
            res=self.query(image,text_prompt)
            print('res',res)
            data_sample[item['BehaviorID']].append({"test_case":test_cases[item["BehaviorID"]],"generation":res})
            # data_sample[item['BehaviorID']]=[data_sample[item['BehaviorID']]]
            i+=1
        save_dir=os.path.join('/home/yjj/MLLM-Jailbreak-evaluation-MMJ-Bench/test_cases',attack_method)
        os.makedirs(save_dir,exist_ok=True)
        with open(os.path.join(save_dir,f'{self.name}_adashield_generation_{attack_method}_test_cases.json'),'w') as save_file:
            json.dump(data_sample,save_file,indent=4)
        data.close()


    def compute_loss(self, behavior, target, image_input):
        conv_prompt = f"<image>\nUSER: {behavior}\nASSISTANT: {target}"
        text_inputs = self.processor.tokenizer(conv_prompt, return_tensors='pt', **self.tokenizer_kwargs).to(self.device)
        pixel_values = self.normalize(image_input).unsqueeze(0).to(self.device)
        inputs = BatchFeature(data={**text_inputs, "pixel_values": pixel_values})
        model_inputs = self.model.prepare_inputs_for_generation(**inputs)

        target_ids = self.processor.tokenizer(target, return_tensors='pt', add_special_tokens=False, **self.tokenizer_kwargs)['input_ids'].to(self.device)
        assert text_inputs['input_ids'][0][-len(target_ids[0]):].equal(target_ids[0]), "Text inputs do not end with target ids"

        outputs = self.model(**model_inputs)
        shift_logits = outputs.logits[0, -(target_ids.size(1)+1):-1, :]
        loss_fn = CrossEntropyLoss()
        return loss_fn(shift_logits, target_ids.view(-1))

    def compute_loss_batch(self, x_adv, batch_targets):
        pass

class LLaVA_v1_6(MultiModalModel):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LlavaNextForConditionalGeneration.from_pretrained('/home/wfh/models/llava-v1.6-vincuna-7b-hf').to(self.device)
        self.processor = LlavaNextProcessor.from_pretrained('/home/wfh/models/llava-v1.6-vincuna-7b-hf')

    def generate(self, test_cases, image_dir, **generation_kwargs):
        generation_kwargs.setdefault('do_sample', False)
        generation_kwargs.setdefault('num_beams', 1)
        assert generation_kwargs['do_sample'] is False, "do_sample should be False"
        assert generation_kwargs['num_beams'] == 1, "num_beams should be 1"
        outputs = []
        for case in test_cases:
            image, prompt = case
            if isinstance(image, str):
                raw_image = Image.open(os.path.join(image_dir, image))
                output = self.query(raw_image, prompt)
                outputs.append(output)

        return outputs 

    def query(self, image, prompt):
        text = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:"
        inputs = self.processor(text=text, images = image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=100)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        answer = answer.split('ASSISTANT:')[-1].strip()
        return answer

    
    def compute_loss(self, behavior, target, image_input):
        return super().compute_loss(behavior, target, image_input)
    
    def compute_loss_batch(self, x_adv, batch_targets):
        pass