import os, ssl, smtplib
from email.message import EmailMessage
from typing import Sequence, Tuple


def send_trade_signal_email(symbol_decisions: Sequence[Tuple[str, str]]) -> None:
    """Send one or more trading signal emails using Gmail SMTP.

    Args:
        symbol_decisions: Iterable of ``(symbol, decision)`` pairs to include in the
            notification email. If empty, no email is sent.

    Environment variables required:
    - GMAIL_USER: Gmail address used as sender
    - GMAIL_APP_PW: 16-character app password
    - GMAIL_TO: fixed recipient address
    """

    if not symbol_decisions:
        return

    gmail_user = os.environ["GMAIL_USER"]
    gmail_app_pw = os.environ["GMAIL_APP_PW"]

    to_email = "zhanghantao91@gmail.com"

    if len(symbol_decisions) == 1:
        symbol, signal = symbol_decisions[0]
        email_subject = f"Trading Signal for {symbol}: {signal}"
    else:
        email_subject = f"Trading Signals Update ({len(symbol_decisions)} tickers)"

    msg = EmailMessage()
    msg["Subject"] = email_subject
    msg["From"] = gmail_user
    msg["To"] = to_email

    text_lines = ["Trading signal notification", ""]
    for symbol, signal in symbol_decisions:
        text_lines.append(f"Symbol: {symbol}")
        text_lines.append(f"Signal: {signal}")
        text_lines.append("")
    text_body = "\n".join(text_lines).rstrip() + "\n"

    html_items = "".join(
        f"<li><b>{symbol}</b>: {signal}</li>" for symbol, signal in symbol_decisions
    )
    html_body = f"""
<html>
  <body>
    <p><b>Trading signal notification</b></p>
    <ul>
      {html_items}
    </ul>
  </body>
</html>
"""

    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(gmail_user, gmail_app_pw)
        smtp.send_message(msg)

