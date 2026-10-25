# from pathlib import Path
# from dotenv import load_dotenv
# import os
# import psycopg
# from sqlalchemy import create_engine, text
# from sqlalchemy.engine import URL

# BASE_DIR = Path(__file__).resolve().parent
# ENV_PATH = BASE_DIR / ".env"

# load_dotenv(dotenv_path=ENV_PATH, override=True)

# DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
# DB_PORT = int(os.getenv("DB_PORT", "5432"))
# DB_NAME = os.getenv("DB_NAME", "marketing_ai_db")
# DB_USER = os.getenv("DB_USER", "marketing_ai_user")
# DB_PASSWORD = os.getenv("DB_PASSWORD", "12345")

# print("ENV PATH:", ENV_PATH)
# print("DB_HOST:", DB_HOST)
# print("DB_PORT:", DB_PORT)
# print("DB_NAME:", DB_NAME)
# print("DB_USER:", DB_USER)
# print("DB_PASSWORD:", DB_PASSWORD)

# print("\n--- Testing psycopg direct connection ---")

# try:
#     with psycopg.connect(
#         host=DB_HOST,
#         port=DB_PORT,
#         dbname=DB_NAME,
#         user=DB_USER,
#         password=DB_PASSWORD,
#     ) as conn:
#         with conn.cursor() as cur:
#             cur.execute("SELECT current_database(), current_user, inet_server_port();")
#             result = cur.fetchone()
#             print("PSYCOPG CONNECTED SUCCESSFULLY:", result)

# except Exception as e:
#     print("PSYCOPG CONNECTION FAILED:")
#     print(e)


# print("\n--- Testing SQLAlchemy connection ---")

# try:
#     database_url = URL.create(
#         drivername="postgresql+psycopg",
#         username=DB_USER,
#         password=DB_PASSWORD,
#         host=DB_HOST,
#         port=DB_PORT,
#         database=DB_NAME,
#     )

#     engine = create_engine(database_url)

#     with engine.connect() as conn:
#         result = conn.execute(
#             text("SELECT current_database(), current_user, inet_server_port();")
#         ).fetchone()

#         print("SQLALCHEMY CONNECTED SUCCESSFULLY:", result)

# except Exception as e:
#     print("SQLALCHEMY CONNECTION FAILED:")
#     print(e)