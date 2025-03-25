import sqlite3 as sl
import datetime
import random
import os, sys
import pathlib
from pathlib import Path

from aiogram import Bot, Dispatcher, types, F, Router
from aiogram.types import Message, ReplyKeyboardRemove, FSInputFile, File, CallbackQuery, PollOption
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import StatesGroup, State
from aiogram.utils.keyboard import InlineKeyboardBuilder, CallbackData
import pathlib
from pathlib import Path
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

router = Router()

def restart():
        import sys
        print("argv was",sys.argv)
        print("sys.executable was", sys.executable)
        print("restart now")

        import os
        os.execv(sys.executable, ['python'] + sys.argv)

back = ReplyKeyboardBuilder()
for item in ('Назад',):
    back.add(types.KeyboardButton(text=item))
back.adjust(1)


# открываем файл с базой данных
bdfile = Path(pathlib.Path.home(), 'GitHub', 'Neuro_timbot', 'neuro_timbot.db')
con = sl.connect(bdfile, check_same_thread=False)
cur = con.cursor()

cur.execute("""CREATE TABLE IF NOT EXISTS parameters(
    id INTEGER primary key AUTOINCREMENT,
    param TEXT,
    value TEXT,
    default_ TEXT,
    mark TEXT);
""")
con.commit()

'''
if cur.execute('SELECT * FROM parameters WHERE param = ?', ('sys_promt',)).fetchone() != []:
    sys_promt = 'Ты большая и умная языковая модель, твоя цель - помогать и отвечать, ответ давай всегда на русском языке. \
    Что на изображении, дай отчет, ответь на русском языке'

    cur.execute('INSERT INTO parameters (param, value, default_, mark) VALUES(?, ?, ?, ?)', 
                ('sys_promt', sys_promt, sys_promt, ''))
    
    promt = 'нужно отвечать очень подробно, описывая все детали, в дальнейшем твой отчет пойдет на оценку.'
    cur.execute('INSERT INTO parameters (param, value, default_, mark) VALUES(?, ?, ?, ?)', 
                ('promt', promt, promt, ''))
    
    con.commit()
'''


#################################
###### L O A D # M O D E L ######
#################################

# default: Load the model on the available device(s)
#model = Qwen2VLForConditionalGeneration.from_pretrained(
#    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")

model_name = 'Qwen2.5-VL-32B-Instruct-bnb-4bit'

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")

def wat(filename, txt, max_new_tokens_input = 128):
    filepath = '~/GitHub/Neuro_timbot/in/'
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": filepath + filename,
                },
                {"type": "text", "text": txt},
            ],
        }
    ]
    # default processer
    #processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained(model_name)


    # The default range for the number of visual tokens per image in the model is 4-16384. 
    # You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens_input)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text
    
@router.message(Command("r1131"))
async def reg1(message: types.Message, state: FSMContext):
    await message.answer('Всёпака!')
    restart()

@router.message(Command("u1131"))
async def update1(message: types.Message, state: FSMContext):
    os.system('/home/heural/GitHub/Neuro_timbot/update')
    await message.answer('Go update yourself first! iam ok')

@router.message(F.photo)
#@router.message(Command("start"))
async def stat(message: types.Message, state: FSMContext, bot: Bot):
    try:
        filepath = '~/GitHub/Neuro_timbot/in/'
        if not os.path.exists(filepath): 
            os.makedirs(filepath) 

        if message.photo != None:
            #photos = message.photo
            #for photo in photos:
            try:
                photo = message.photo[3]
            except:
                try:
                    photo = message.photo[2]
                except:
                    try:
                        photo = message.photo[1]
                    except:
                        photo = message.photo[0]
            photo_file = await bot.get_file(photo.file_id)
            photo_path = photo_file.file_path
            filename = str(message.from_user.id) + '_' + str(random.randrange(1, 1000000)) + '.jpg'
            await bot.download_file(photo_path, filepath + filename)
            status = True
        elif message.document != None:
            doc = message.document
            doc_file = await bot.get_file(doc.file_id)
            doc_path = doc_file.file_path
            ext = str(doc_path).split('.')
            filename = filename[:-3] + ext[-1]
            await bot.download_file(doc_path, filepath + filename)
            #await message.answer(f'{ext[-1]}')
            status = True
        else:
            print('Ащельме! На проверку прислали фиг знает что')
            await message.answer('Что-то не то с файлом')
            status = False
        if status:
            txt_sys = cur.execute('SELECT value FROM parameters WHERE param = ?', ('sys_promt',)).fetchone()[0]
            txt_promt = cur.execute('SELECT value FROM parameters WHERE param = ?', ('promt',)).fetchone()[0]

            txt = txt_sys + txt_promt
            
            answer = wat(filename, txt, 512)
            print(answer)
            #print('txt =', type(txt))
            #print('output_text =', type(output_text))
            #await message.answer(txt)
            await message.answer(str(answer[0]))

    except Exception as error: 
        await message.answer(f'{error}')
        print(f'{error}')
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f'{exc_type, fname, exc_tb.tb_lineno}')
        await message.answer(f'{exc_type, fname, exc_tb.tb_lineno}')



@router.message(F.text == 'Промт')
@router.message(Command("промт"))
async def back(message: types.Message, state: FSMContext):
    pass