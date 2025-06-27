from collections import Counter
import pandas as pd 
import math

stop_words_list = ['000', '007', '07th', '09',
                       '10', '100', '101', '1047', 
                        '11', '12', '120', '13', '130cm',
                       '13am', '13th', '14', '141', '15',
                       '1500', '16', '17', '18', '180', '1939',
                       '1942', '1960s', '1980s', '1990s',
                       '1995', '1996', '1997', '1998', '1999',
                       '19th', '1c', '1soft', '1st', '20', '2000',
                       '2001','2002', '2003', '2004', '2005', '2006',
                       '2007', '2008', '2009', '2010', '2011', '2012',
                       '2013', '2014', '2015', '2016', '2017', '2018',
                       '2019', '2020', '2020venture',  '20th', '21',
                       '21st', '22', '221b', '227',  '22nd', '23rd', '24',
                       '258', '285',  '2darray',  '2nd', '2x2',
                       '2xl', '3000', '3000ad', '32x', '343',
                       '35', '358', '360', '369', '3a', '3d6',
                       '3ds', '3g', '3lv', '3rd', '3x3', '40',
                       '400', '44', '45', '46', '4a', '4bit', 
                       '4head', '4j', '4sdk', '4x', '50th',
                       '51', '5200', '562', '5656', '59',  '5d', '5pb',
                       '5th', '60', '6010', '64', '6e6e6e', '76', '777',
                       '777next', '7dfps', '7th', '800', '82', '8888888',
                       '88mm', '8floor', '8monkey', '8th', '935', '98',
                       '98demake', '9heads', '9th', 'a2a', 'a2z',
                       '0verflow', '10kbit', '10tacle', '10tons', '2049er', '20xx',
                       '22cans', '2awesome', '2bad', '2d', '2dengine', '2dogs', '2k',
                       '34bigthings', '3d', '3dclouds', '3division', '3do', '3drunkmen',
                       '3rdeye', '3vision', '3vr', '49ers', '49games', '4d', '4fufelz',
                       '4gency', '5bit', 'aaa', 'aaaaaaaaaaaaaaaaaaaaaaaaa', 'ab',
                       'e10', 'e3', 'e404', 'pax',
                       'achievements', 'com', 'comachievements', 'companynameintitle',
                       'crowdfunded', 'declarativetitle', 'digitaldistribution',
                       'e32005', 'e32007', 'e32008', 'e32009', 'e32010',
                       'e32011', 'e32012', 'e32013', 'e32014', 'e32015',
                       'e32016', 'e32017', 'e32018', 'e32019', 'e32020',
                       'easyanticheat', 'epicgamesstore', 'gametitlesthatarealsoquestions', 'gog',
                       'humblebundle', 'kickstarterfunded', 'licensedgame', 'onlive',
                       'paxeast2005', 'paxeast2007', 'paxeast2008', 'paxeast2009',
                       'paxeast2010', 'paxeast2011', 'paxeast2012', 'paxeast2013',
                       'paxeast2014', 'paxeast2015', 'paxeast2016', 'paxeast2017',
                       'paxeast2018', 'paxeast2019', 'paxeast2020',
                       'paxprime', 'paxprime2005', 'paxprime2007', 'paxprime2008',
                       'paxprime2009', 'paxprime2010', 'paxprime2011', 'paxprime2012', 'paxprime2013',
                       'paxprime2014', 'paxprime2015', 'paxprime2016', 'paxprime2017', 'paxprime2018',
                       'paxprime2019', 'paxprime2020', 'paxsouth2005', 'paxsouth2007', 'paxsouth2008',
                       'paxsouth2009', 'paxsouth2010', 'paxsouth2011', 'paxsouth2012', 'paxsouth2013',
                       'paxsouth2014', 'paxsouth2015', 'paxsouth2016', 'paxsouth2017', 'paxsouth2018',
                       'paxsouth2019', 'paxsouth2020',
                       'paxwest2005', 'paxwest2007', 'paxwest2008', 'paxwest2009',
                       'paxwest2010', 'paxwest2011', 'paxwest2012', 'paxwest2013',
                       'paxwest2014', 'paxwest2015', 'paxwest2016', 'paxwest2017',
                       'paxwest2018', 'paxwest2019', 'paxwest2020', 'playstation',
                       'playstationplus', 'playstationtrophies', 'realphotosoncoverart',
                       'secretachievements', 'smartdelivery', 'steam', 'steamapplearcade',
                       'steamcloud', 'steamgreenlight', 'steamremoteplaytogether', 'steamtradingcards',
                       'steamturnnotifications', 'threewordgametitlewithconjunctionorpreposition',
                       'trophies', 'valveindexsupport', 'xboxonexenhanced', 'xboxplayanywhere']

def compute_tf(doc):
    tf_dict = Counter(doc)
    total_terms = len(doc)
    return {term: count/total_terms for term, count in tf_dict.items()}

def compute_idf(docs):
    N = len(docs) 
    idf_dict = {}
    all_terms = set(term for doc in docs for term in doc)
    for term in all_terms:
        df = sum(1 for doc in docs if term in doc)
        idf_dict[term] = math.log((1+N)/((1+df)+1))
    return idf_dict

def tf_idf(tf, idf):
    return {term: tf_val * idf[term] for term, tf_val in tf.items() if term in idf}

def tfidf_func(docs, stopwords=None  ,max_features = 1250):

    if stopwords is None:
        stopwords = set()
    # Tách từ, chuyển thành list và loại bỏ stopword
    tokenized_docs = []
    for doc in docs:
        tokens = doc.lower().split()
        filtered_tokens = [t for t in tokens if t not in stopwords]
        tokenized_docs.append(filtered_tokens)
    
    # Tính tf cho mỗi văn bản
    tf_list = [compute_tf(doc) for doc in tokenized_docs]
    # Tính idf trên toàn tập văn bản
    idf =  compute_idf(tokenized_docs)
    # if-idf
    tfidf_list = [tf_idf(tf, idf) for tf in tf_list]
    # Chuyển thành dataframe
    all_terms = sorted(idf.keys())
    # khởi tạo ma trận ban đầu
    tfidf_matrix = pd.DataFrame([
        [doc.get(term,0.0) for term in all_terms] for doc in tfidf_list],
        columns=all_terms)
    top_terms = tfidf_matrix.sum(axis=0).sort_values(ascending=False).head(max_features).index
    filtered_idf = {term: idf[term] for term in top_terms}
    return tfidf_matrix[top_terms],filtered_idf

