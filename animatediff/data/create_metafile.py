import  os

# folder_path = '/dataset00/Videos/smile'
# txt_file_path = '/dataset00/Videos/smile/filelist.txt'

# with open(txt_file_path, 'w') as txt_file:
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.gif'):
#             txt_file.write(file_name + '\n')

# with open(txt_file_path, 'r') as txt_file:
#     lines = txt_file.readlines()
#     lines = [x.strip() for x in lines]
#     print(lines)

folder_path = '/dataset00/Videos/CelebV-Text/videos/celebvtext_6'
txt_file_path = '/dataset00/Videos/CelebV-Text/descripitions/filelist.txt'

# with open(txt_file_path, 'w') as txt_file:
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.mp4'):
#             txt_file.write(file_name + '\n')

with open(txt_file_path, 'r') as txt_file:
    lines = txt_file.readlines()[:]
    lines = [x.strip() for x in lines]
    # print(lines)
happy_count = 0

happy_txt_file_path = '/dataset00/Videos/CelebV-Text/descripitions/happy_filelist_noturn.txt'
with open(happy_txt_file_path, 'w') as happy_txt_file:
    for line in lines:
        video_path = os.path.join(folder_path, line)
        emotion_desc_filepath = video_path.replace('videos/celebvtext_6','descripitions/emotion').replace('.mp4','.txt')
        try:
            with open(emotion_desc_filepath, 'r') as txt_file:
                lines = txt_file.readlines()
                # lines = [x.strip() for x in lines]
                line0 = lines[0].strip()
                if 'happy' in line0 and len(line0.split(',')) < 2:
                    happy_count += 1
                    happy_txt_file.write(line + '\n')
                    print(line0)

        except:
            continue
print('happy caption num is ', happy_count)



    