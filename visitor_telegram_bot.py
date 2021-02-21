from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters

import os

token = '1676520849:AAG7Vx2QE4XBQJPKrnZotdeKhjoAhZDJufQ'

print('start telegram chat bot')

def get_message(update, context):
    message = update.message.text
    if len(message) == 16:
        year, month, day, start_hour, end_hour = int(message[:4]), int(message[5:7]), int(message[8:10]), int(message[11:13]), int(message[14:16])
        total_faces = 0
        total_visitors = 0
        
        while start_hour != end_hour:
            filename = "%04d.%02d.%02d.%02d-%02d.txt" % (year, month, day, start_hour, start_hour + 1)
        
            f = open(filename, 'r')

            while True:
                line = f.readline()

                if not line : break

                if line[:32] == 'total number of unknown faces : ':
                    total_faces += int(line[32:])
                elif line[:27] == 'total number of visitors : ':
                    total_visitors += int(line[27:])

            f.close()
            start_hour += 1

        print('start reply')
        update.message.reply_text("total number of unknown faces : {}\ntotal number of visitors : {}".format(total_faces, total_visitors))

def help_command(update, context) :
    update.message.reply_text("You can view the statics when you enter <YYYY.MM.DD.HH-HH> format")


updater = Updater(token, use_context=True)

message_handler = MessageHandler(Filters.text & (~Filters.command), get_message)
updater.dispatcher.add_handler(message_handler)

help_handler = CommandHandler('help', help_command)
updater.dispatcher.add_handler(help_handler)

updater.start_polling(timeout=3, clean=True)
updater.idle()