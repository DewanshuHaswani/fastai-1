from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram.ext.dispatcher import run_async
import logging
import os
import sys
from dog_breed_model import Model

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

BASE_FILE_PATH = '/tmp/{}.jpg'

model = Model()

def help(bot, update):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Загрузи фото песик - я скажу какой это тип!')


def echo(bot, update):
    """Echo the user message."""
    update.message.reply_text('Загрузи фото песик - я скажу какой это тип!')

@run_async    
def process_photo(bot, update):
    message = update.message
    file_path = BASE_FILE_PATH.format(message.message_id)
    photo = message.photo[-1]
    update.message.reply_text('Думаю..')
    dog_breed = 'К сожалению не получилось определить'
    try:
        photo_file = photo.bot.get_file(photo.file_id)
        photo_file.download(file_path)
        dog_breed = 'Кажется это '+ model.identify_dog_breed(file_path)
    finally:
        os.remove(file_path)
        
    update.message.reply_text(dog_breed)

def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    """Start the bot."""
    # Create the EventHandler and pass it your bot's token.
    updater = Updater("526570426:AAF-1S5qfM7gmXFa6216LiMJhpgMrSVPPGU")

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(~Filters.photo, echo))
    
    dp.add_handler(MessageHandler(Filters.photo, process_photo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()