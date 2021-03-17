import pandas as pd
import re
import numpy as np

def main() :
    preprocess_user_train()
    preprocess_product()
    preprocess_user_test()


def preprocess_user_train() :

    chunksize = 10000
    list_of_dataframes = []
    for df in pd.read_csv('./recommendation_dataset/pred_train_anon.dat',
                header=None, sep='[|]', chunksize=chunksize,engine='python'):
        # process your data frame here
        # then add the current data frame into the list
        list_of_dataframes.append(df)

    # if you want all the dataframes together, here it is
    df1 = pd.concat(list_of_dataframes)

    df2 = df1.reset_index(drop=True)
    df2.columns = ['user_id','product_id']

    df3 = df2.product_id.str.replace(':',',')
    df4 = df3.str.split(',')
    df2.product_id = df4

    total_date = []
    for j in range(len(df2)):
        date = []
        for i in df2.product_id[j] :
            if i.startswith('2018') :
                date.append(i)
                df2.product_id[j].remove(i)
        total_date.append(date)
        
    v = np.column_stack([total_date, df2.user_id.values])
    c = ['data'.format(i) for i in range(v.shape[1] - 1)] + ['user_id']
    df3 = pd.DataFrame(v, df2.index, c)
    df4 = pd.merge(df2,df3, on = 'user_id')
    user = df4.apply(pd.Series.explode)

    user.to_csv('../data/user_train.csv')

def preprocess_product() :

    df = pd.read_csv('./recommendation_dataset/products_anon.dat',error_bad_lines=False)
    df.columns = ['user_id','product_id','price','low_price']

    
    p = re.compile('[\!@#$%\^&\*\(\)\-\=\[\]\{\}\.,/\?~\+\'"|_:;><`┃name|id|price|low_price]')

    def remove_special_characters(sentence, lower = True) :
        sentence = p.sub(' ', sentence)
        sentence = ' '.join(sentence.split())
        
        if lower :
            sentence = sentence.lower()
            
        return sentence

    for col in df.columns :
        df[col] = df[col].map(remove_special_characters)
    df.to_csv('../data/product.csv')

def preprocess_user_test() :

    chunksize = 10000
    list_of_dataframes = []
    for df in pd.read_csv('./recommendation_dataset/pred_test_anon.dat',
                header=None, sep='[|]', chunksize=chunksize,engine='python'):
        # process your data frame here
        # then add the current data frame into the list
        list_of_dataframes.append(df)

    # if you want all the dataframes together, here it is
    df1 = pd.concat(list_of_dataframes)

    df2 = df1.reset_index(drop=True)
    df2.columns = ['user_id','product_id']

    df3 = df2.product_id.str.replace(':',',')
    df4 = df3.str.split(',')
    df2.product_id = df4

    total_date = []
    for j in range(len(df2)):
        date = []
        for i in df2.product_id[j] :
            if i.startswith('2018') :
                date.append(i)
                df2.product_id[j].remove(i)
        total_date.append(date)
        
    v = np.column_stack([total_date, df2.user_id.values])
    c = ['data'.format(i) for i in range(v.shape[1] - 1)] + ['user_id']
    df3 = pd.DataFrame(v, df2.index, c)
    df4 = pd.merge(df2,df3, on = 'user_id')
    user = df4.apply(pd.Series.explode)

    user.to_csv('../data/user_train.csv')



if __name__ == '__main__':
    main()