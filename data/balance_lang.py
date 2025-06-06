import pandas as pd

df = pd.read_csv('spam_enhanced_labeled.csv', encoding='utf-8-sig')

def balance(df, lang):
    sub = df[df['lang'] == lang]
    n_spam = (sub['label'] == 'spam').sum()
    n_ham = (sub['label'] == 'ham').sum()
    n = min(n_spam, n_ham)
    spam_df = sub[sub['label'] == 'spam'].sample(n, random_state=42)
    ham_df = sub[sub['label'] == 'ham'].sample(n, random_state=42)
    balanced = pd.concat([spam_df, ham_df], ignore_index=True)
    print(f'{lang} - spam:{n_spam}, ham:{n_ham}, used:{n} each')
    return balanced

df_en = balance(df, 'english')
df_cn = balance(df, 'chinese')

df_en.to_csv('en_balanced.csv', index=False, encoding='utf-8-sig')
df_cn.to_csv('cn_balanced.csv', index=False, encoding='utf-8-sig')
print('已生成均衡采样后的 data/en_balanced.csv 及 data/cn_balanced.csv')