import asyncio

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from core.config import settings
from core.security import create_email_token


async def send_verification_email(email: str) -> None:
    token = create_email_token(email)
    verify_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"

    if not settings.SENDGRID_API_KEY:
        return  # dev env — skip actual sending

    message = Mail(
        from_email=settings.FROM_EMAIL,
        to_emails=email,
        subject="StockSenseAI 이메일 인증",
        html_content=(
            f"<p>아래 링크를 클릭하여 이메일을 인증하세요 (30분 유효):</p>"
            f'<a href="{verify_url}">{verify_url}</a>'
        ),
    )
    try:
        client = SendGridAPIClient(settings.SENDGRID_API_KEY)
        await asyncio.to_thread(client.send, message)
    except Exception:
        pass  # log in production; don't fail registration if email send fails
