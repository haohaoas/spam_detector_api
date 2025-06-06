import imaplib
import email
from email.header import decode_header
import pandas as pd

# QQ邮箱相关信息
IMAP_SERVER = 'imap.qq.com'
EMAIL_ACCOUNT = '1339239726@qq.com'
EMAIL_PASSWORD = 'upxpqxcxfcjybafa'  # 注意不是QQ密码，是授权码

def clean_text(text):
    # 只留可显示内容，去除多余空格
    return "".join(text.replace('\r','').replace('\n','').split())

# 登录邮箱
mail = imaplib.IMAP4_SSL(IMAP_SERVER)
mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
mail.select('INBOX')

# 拉取最近200封邮件
typ, data = mail.search(None, 'ALL')
mail_ids = data[0].split()
mail_ids = mail_ids[-200:]  # 最新200封

ham_texts = []

for num in mail_ids:
    typ, msg_data = mail.fetch(num, '(RFC822)')
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            msg = email.message_from_bytes(response_part[1])
            # 标题
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding if encoding else "utf-8", errors='ignore')
            # 正文
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        charset = part.get_content_charset()
                        body = part.get_payload(decode=True)
                        if charset:
                            try:
                                body = body.decode(charset, errors='ignore')
                            except:
                                body = body.decode('utf-8', errors='ignore')
                        else:
                            body = body.decode('utf-8', errors='ignore')
                        break
            else:
                body = msg.get_payload(decode=True)
                try:
                    body = body.decode('utf-8', errors='ignore')
                except:
                    body = str(body)
            # 合成短文本（截断，只留前200字防止泄露隐私）
            sample_text = clean_text(subject) + '。' + clean_text(body)[:200]
            ham_texts.append({'label':'ham','text':sample_text})

mail.logout()

# 保存到csv
df = pd.DataFrame(ham_texts)
df.to_csv('../../data/qqmail_ham.csv', index=False, encoding='utf-8-sig')
print('导出', len(df), '封邮件为ham样本: data/qqmail_ham.csv')