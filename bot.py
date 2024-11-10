import asyncio
import logging

from dataclasses import dataclass
from environs import Env
import sqlite3 as sl
from aiogram import Bot, Dispatcher
from handlers import common
import pathlib
from pathlib import Path

print('  >>> Telegram bot started')

# Создаем экземпляр класса Env
env: Env = Env()

# Добавляем в переменные окружения данные, прочитанные из файла .env 
env.read_env()

@dataclass
class DatabaseConfig:
    database: str         # Название базы данных
    db_host: str          # URL-адрес базы данных
    db_user: str          # Username пользователя базы данных
    db_password: str      # Пароль к базе данных


@dataclass
class TgBot:
    token: str            # Токен для доступа к телеграм-боту
    admin_ids: list[int]  # Список id администраторов бота


@dataclass
class Config:
    tg_bot: TgBot
    db: DatabaseConfig

# Создаем экземпляр класса Config и наполняем его данными из переменных окружения
config = Config(
    tg_bot=TgBot(
        token=env('BOT_TOKEN'),
        admin_ids=list(map(int, env.list('ADMIN_IDS')))
    ),
    db=DatabaseConfig(
        database=env('DATABASE'),
        db_host=env('DB_HOST'),
        db_user=env('DB_USER'),
        db_password=env('DB_PASSWORD')
    )
)

bdfile = Path(pathlib.Path.home(), 'GitHub', 'Neuro_timbot', 'neuro_timbot.db')
con = sl.connect(bdfile, check_same_thread=False)
cur = con.cursor()

# потом сделать - кто первый зашел в бота - тот и админ

cur.execute("""CREATE TABLE IF NOT EXISTS logging(
    id INTEGER primary key AUTOINCREMENT,
    userid INTEGER,
    username TEXT,
    nick TEXT,
    filepath TEXT,
    response TEXT,
    date_ TEXT,
    mark TEXT);
""")
con.commit()

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    dp = Dispatcher()
    bot = Bot(token=config.tg_bot.token)

    dp.include_router(common.router)

    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
