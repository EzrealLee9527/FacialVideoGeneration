from omegaconf import OmegaConf
import os
import argparse
def main(args):
    # sticker_templates=os.listdir(args.stickers_path)

    #lora:1)对应lora模型，2）对应lora_id 3)gender
    # prompt:1)基于gender提供prompt，两个gender各给4个通用prompt
    # template:每个template提供一个这个表情包准确的prompt

    #要改的：lora_id, lora_path,prompt(根据gender和template来选)

    #prompt config
    gender_prompt_config={}
    prompt_config=OmegaConf.load("prompts_by_gender.yaml")
    for gender,gender_config in list(prompt_config.items()):
        gender_prompt_config[gender]=gender_config
        # print(len(gender_config.prompt))
    
    #lora config
    lora_config=OmegaConf.load("loras.yaml")
    # for lora_id,lora_config in list(lora_config.items()):
        # print(lora_id)
        # print(lora_config.gender)
        # print(lora_config.lora_path)
        # print(os.path.exists(lora_config.lora_path))
    
    #template config
    template_configs=OmegaConf.load("stickers_templates.yaml")

    for lora_id,lora_config in list(lora_config.items()):
        for tempate,template_config in list(template_configs.items()):

            config=OmegaConf.load("configs/prompts/mj_girl_id38_sticker_result_emma.yaml")
            
            #set lora path and lora_id to get lora model and lora-reconstructed deca labels
            config["FilmVelvia"].path=lora_config.lora_path
            config["FilmVelvia"].lora_id=lora_id
            
            #set seed to -1 to get random seeds
            config["FilmVelvia"].seed=-1

            #set new prompts
            prompt=list(gender_prompt_config[lora_config.gender].prompt)
            n_prompt=list(gender_prompt_config[lora_config.gender].n_prompt)

            #replace prompt with gender
            if lora_config.gender=="female":
                # print(template_config.prompt.replace("person","girl"))
                # print(len(prompt))
                # print(template_config)
                prompt.extend([str(template_config.prompt.replace("person","girl"))])
                n_prompt.extend([str(template_config.n_prompt)])
            elif lora_config.gender=="male":
                # print("prompt:{}".format(prompt))
                prompt.extend([str(template_config.prompt.replace("person","man"))])
                # print("add prompt:{}".format(template_config.prompt.replace("person","man")))
                n_prompt.extend([str(template_config.n_prompt)])
            
            config["FilmVelvia"].prompt=prompt
            config["FilmVelvia"].n_prompt=n_prompt
            

            #set template params
            config["FilmVelvia"].video_id=template_config.video_id
            config["FilmVelvia"].start_f_idx=template_config.start_f_idx
            config["FilmVelvia"].video_length=template_config.video_length
            config["FilmVelvia"].frame_stride=template_config.frame_stride

            config["FilmVelvia"].W=template_config.W

            #save_new config
            OmegaConf.save(config,"template_lora_configs/{}_{}.yaml".format(lora_id,template_config.video_id))
            
            # print(template_config.prompt)
            # print(template_config.n_prompt)



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    # parser.add_argument("--stickers_path",required=True,type=str)
    args=parser.parse_args()
    main(args)
