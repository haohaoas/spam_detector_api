import pandas as pd

df_main = pd.read_csv('spam_enhanced_labeled.csv', encoding='utf-8-sig')
df_qqmail = pd.read_csv('qqmail_ham.csv', encoding='utf-8-sig')

# 如果qqmail_ham没有'lang'列，需要加上
df_qqmail['lang'] = 'chinese'

df_all = pd.concat([df_main, df_qqmail], ignore_index=True)
df_all.to_csv('spam_enhanced_labeled.csv', index=False, encoding='utf-8-sig')
print(f"已合并，合计 {len(df_all)} 条，存为 spam_enhanced_labeled.csv")