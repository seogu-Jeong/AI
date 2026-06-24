import os

from celery import Celery
from celery.schedules import crontab

celery_app = Celery(
    "tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    include=["tasks.ai_tasks", "tasks.auto_trade_tasks", "tasks.email_tasks", "tasks.order_tasks"],
)

celery_app.conf.beat_schedule = {
    "refresh-ai-signals": {
        "task": "tasks.ai_tasks.refresh_ai_signals",
        "schedule": crontab(hour=15, minute=35, day_of_week="mon-fri"),
    },
    "check-price-alerts": {
        "task": "tasks.email_tasks.check_price_alerts",
        "schedule": 300.0,
    },
    "check-daily-loss": {
        "task": "tasks.email_tasks.check_daily_loss",
        "schedule": 600.0,
    },
}
celery_app.conf.timezone = "Asia/Seoul"
celery_app.conf.broker_connection_retry_on_startup = True
