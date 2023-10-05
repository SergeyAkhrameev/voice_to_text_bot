import os
import time
import numpy as np

import telebot as tb
import whisper_timestamped as whisper
import ffmpeg
import json
import torch
import math

from pydub import AudioSegment

#this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

#help(whisper.transcribe)

# Замените "YOUR_BOT_TOKEN" на токен вашего бота
bot = tb.TeleBot('')

# Обработчик голосовых сообщений
@bot.message_handler(content_types=["voice"])
def handle_voice_message(message):
    try:
        # Получаем голосовое сообщение
        voice = bot.get_file(message.voice.file_id)
        voice_info = bot.get_file(message.voice.file_id)
        voice_file = bot.download_file(voice.file_path)

        # Сохраняем голосовое сообщение в файл
        voice_file_path_ogg = os.path.join("voice_messages", f"{message.message_id}.ogg")
        with open(voice_file_path_ogg, "wb") as voice_file_local:
            voice_file_local.write(voice_file)

        # Преобразуем .ogg в .wav
        voice_file_path_wav = os.path.join("voice_messages", f"{message.message_id}.wav")
        audio = AudioSegment.from_ogg(voice_file_path_ogg)
        audio = audio.set_frame_rate(16000).set_sample_width(2)  # Установите желаемую частоту дискретизации и ширину выборки
        audio.export(voice_file_path_wav, format="wav")

        #voice_file_path_wav ='voice_messages/respondent.wav' #Load file directly

        print(f"voice_file_path_wav: {voice_file_path_wav}")

        # Распознавание речи с помощью WhisperTimestamped
        audio = whisper.load_audio(voice_file_path_wav)
        model = whisper.load_model("large", device='cpu')
        
        start_time = time.time()
        result = whisper.transcribe(model, audio, language="ru")
        end_time = time.time()

        elapsed_time = end_time - start_time
        

        #print(torch.device)



        '''
        audio_list=[]
        for i in range(len(result['text']['segments'][i])):
            audio_data = dict(audio_id = result['text']['segments'][i]['start']['text'],)
                
            audio_list.append(audio_data)

            return audio_list
        '''
        
        json_result = json.dumps(result, indent = 2, ensure_ascii = False)
        print(json_result)

        segments =[]
        for item in result['segments']:
            confidence = item['confidence']
            segments.append(confidence)
        
        # Вычислите среднее значение
        if segments:
            average_confidence = np.mean(segments)
            print("Среднее значение уверенности:", average_confidence)
        else:
            print("Список segments пуст.")

        with open("result.txt", "w", encoding="utf-8") as text_file:
            text_file.write(json_result)

        # Максимальная длина сообщения в Telegram
        MAX_MESSAGE_LENGTH = 4000

        

        # Текст, который вы хотите отправить
        text_to_send = result['text']

        if len(text_to_send) <= MAX_MESSAGE_LENGTH:
            bot.send_message(message.chat.id, f"Распознанный текст: {result['text']}")

        else:

            # Разбиваем текст на слова
            words = text_to_send.split()

            # Инициализируем список для хранения частей сообщения
            message_parts = []

            current_part = ""
            for word in words:
                if len(current_part) + len(word) + 1 <= MAX_MESSAGE_LENGTH:
                    # Добавляем слово и пробел к текущей части, если это не приведет к превышению лимита
                    current_part += word + " "
                else:
                    # Если добавление слова приведет к превышению лимита, завершаем текущую часть
                    message_parts.append(current_part.strip())
                    current_part = word + " "

            # Добавляем последнюю часть
            message_parts.append(current_part.strip())

            # Отправляем части сообщения
            for part in message_parts:
                bot.send_message(message.chat.id, f"Распознанный текст: {part}")    


        # Отправляем текстовый ответ
        #bot.send_message(message.chat.id, f"Распознанный текст: {result['text']}")
        #bot.send_message(message.chat.id, f"Распознанный текст: {result['segments'][0]['start']['text']}")
    
        print(f"Время выполнения: {elapsed_time} секунд")
    
    except Exception as e:
        bot.send_message(message.chat.id, f"Произошла ошибка: {str(e)}")

# Запускаем бота
bot.polling(none_stop=True)