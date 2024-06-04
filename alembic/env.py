from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from main import Base  # Замените на путь к вашим моделям

# Эта строка настраивает логирование
config = context.config
fileConfig(config.config_file_name)

# Это MetaData объект из ваших моделей
target_metadata = Base.metadata

# Получаем URL базы данных из alembic.ini
url = config.get_main_option("sqlalchemy.url")

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
