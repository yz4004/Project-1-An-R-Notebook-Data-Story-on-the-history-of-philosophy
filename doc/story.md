# A story about philosophy data

This dataset was compiled for the Philosophy Data Project. It contains over 300,000 sentences from over 50 texts spanning 10 major schools of philosophy, such as Plato, Aristotle, Rationalism, Empiricism, German Idealism, Communism, Capitalism, Phenomenology, Continental Philosophy... The texts were cleaned extensively before being tokenized and organized already. There are 11 features, author,school,... to lemmatized_str. Most of them are strings represents names and text. Only three of them are in int64 format.Publication and edition data are categrical features, and only sentence_length has quantitive meaning. So our analysis is mainly based on text analysis.

First, import some useful packages and read raw data as a pandas dataframe.


```python
import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# import data and see the head to get basic infomation
data = pd.read_csv("philosophy_data1.csv")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>author</th>
      <th>school</th>
      <th>sentence_spacy</th>
      <th>sentence_str</th>
      <th>original_publication_date</th>
      <th>corpus_edition_date</th>
      <th>sentence_length</th>
      <th>sentence_lowered</th>
      <th>tokenized_txt</th>
      <th>lemmatized_str</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>What's new, Socrates, to make you leave your ...</td>
      <td>What's new, Socrates, to make you leave your ...</td>
      <td>-350</td>
      <td>1997</td>
      <td>125</td>
      <td>what's new, socrates, to make you leave your ...</td>
      <td>['what', 'new', 'socrates', 'to', 'make', 'you...</td>
      <td>what be new , Socrates , to make -PRON- lea...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>Surely you are not prosecuting anyone before t...</td>
      <td>Surely you are not prosecuting anyone before t...</td>
      <td>-350</td>
      <td>1997</td>
      <td>69</td>
      <td>surely you are not prosecuting anyone before t...</td>
      <td>['surely', 'you', 'are', 'not', 'prosecuting',...</td>
      <td>surely -PRON- be not prosecute anyone before ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>The Athenians do not call this a prosecution b...</td>
      <td>The Athenians do not call this a prosecution b...</td>
      <td>-350</td>
      <td>1997</td>
      <td>74</td>
      <td>the athenians do not call this a prosecution b...</td>
      <td>['the', 'athenians', 'do', 'not', 'call', 'thi...</td>
      <td>the Athenians do not call this a prosecution ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>What is this you say?</td>
      <td>What is this you say?</td>
      <td>-350</td>
      <td>1997</td>
      <td>21</td>
      <td>what is this you say?</td>
      <td>['what', 'is', 'this', 'you', 'say']</td>
      <td>what be this -PRON- say ?</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>Someone must have indicted you, for you are no...</td>
      <td>Someone must have indicted you, for you are no...</td>
      <td>-350</td>
      <td>1997</td>
      <td>101</td>
      <td>someone must have indicted you, for you are no...</td>
      <td>['someone', 'must', 'have', 'indicted', 'you',...</td>
      <td>someone must have indict -PRON- , for -PRON- ...</td>
    </tr>
  </tbody>
</table>
</div>



Using info() to check the datatype and make sure there is no annoying N/A obstruct data manipulation.


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 360808 entries, 0 to 360807
    Data columns (total 11 columns):
     #   Column                     Non-Null Count   Dtype 
    ---  ------                     --------------   ----- 
     0   title                      360808 non-null  object
     1   author                     360808 non-null  object
     2   school                     360808 non-null  object
     3   sentence_spacy             360808 non-null  object
     4   sentence_str               360808 non-null  object
     5   original_publication_date  360808 non-null  int64 
     6   corpus_edition_date        360808 non-null  int64 
     7   sentence_length            360808 non-null  int64 
     8   sentence_lowered           360808 non-null  object
     9   tokenized_txt              360808 non-null  object
     10  lemmatized_str             360808 non-null  object
    dtypes: int64(3), object(8)
    memory usage: 30.3+ MB
    

Sentence length is the length of sentence string, which comprises space and punctuation.

Tokenized_txt seems to be generated by splitting sentence and removing repeated words. It is the unique words of a sentence. This feature is useful to measure a complexity of a sentence. So I will create a new feature, "n_tokens", represents the number of tokens in a sentence. 


```python
token = list(map(eval, data.tokenized_txt))
data['n_tokens'] = list(map(len, token))
data.head()
```

First, we can look at how many authors does a school have.


```python
# get a list of school
school = data.school.unique()
school, len(school)
```




    (array(['plato', 'aristotle', 'empiricism', 'rationalism', 'analytic',
            'continental', 'phenomenology', 'german_idealism', 'communism',
            'capitalism', 'stoicism', 'nietzsche', 'feminism'], dtype=object),
     13)




```python
# generate a list of array which comprise authors of each school
sch_author = [data[data.school == school[i]].author.unique() for i in range(13)]

# length of array equals to number of authors
num_of_au = [len(sch_author[i]) for i in range(len(sch_author)) ]

fig1 = plt.figure(figsize=(18,5))
plt.bar(school, num_of_au )
```




    <BarContainer object of 13 artists>




    
![png](output_9_1.png)
    


Analytic school has the most authors and plato, aristotle and nietzsche only get one. No strange because they are named by authors themselves.

Check our how many books each author has.


```python
# generate a list comprise author's book title.

au = data.author.unique()
bk_au = [data[data.author == au[i]].title.unique() for i in range(len(au))]
num_of_bk_au = [len(au[i]) for i in range(len(bk_au)) ]
fig2 = plt.figure(figsize=(35,5))
plt.bar(au, num_of_bk_au )
```




    <BarContainer object of 36 artists>




    
![png](output_11_1.png)
    


Next graph shows how many books each author have.


```python
bk_au = data.groupby("author").title.unique().map(len)
fig3 = plt.figure(figsize=(35,5) )
plt.bar(np.sort(au), bk_au )
```




    <BarContainer object of 36 artists>




    
![png](output_13_1.png)
    


Nietzsche has 5 books, which is the most.

But number of books doesn't not represent the real amount of work by author. Aristotle and Plato both have only one text, which is a collection of all their works. We can show the volume size by adding up "sentence_length" as a rough estimate.


```python
vl_au = [data[data.author == au[i]].sentence_length.sum() for i in range(len(au))]
fig4 = plt.figure(figsize=(35,10))
plt.bar(au, vl_au )
```




    <BarContainer object of 36 artists>




    
![png](output_15_1.png)
    



```python
au[np.argmax(vl_au)], au[np.argmin(vl_au)]
```




    ('Aristotle', 'Epictetus')



Among authors, Aristotle has the most of work and Epictetus has the least.

By the same way, check out the volume of works of different schools.


```python
sch_vol = data.groupby("school").sentence_length.sum()
fig5 = plt.figure(figsize=(21,10))
# plt.bar(range(len(school)), sch_vol )
plt.bar(school, sch_vol )
```




    <BarContainer object of 13 artists>




    
![png](output_18_1.png)
    


Plato and Aristotle as the founder of philosophy have a lot of works, and followed by German Idealism and Analyic, which are important genres. 

## Popularity

We want to create a index to measure how popular of a school is.
A complex and vital theory generally is built by many scholars. If a field has few people working on it, whether it is an acient stuff or it is not that popular. So a popular school should have a certain amount of working (here we represents it by total volume, the sum of sentence length) and have many philosophers interested on it. So we use total volume of school times number of author to measure that popularity.


```python
popularity = data.groupby("school").apply(lambda df: df.sentence_length.sum()*len(df.author.unique()) )
popularity = popularity/popularity.sum()  # normalization 

fig6 = plt.figure(figsize=(20,8))
plt.bar(np.sort(school), popularity)
```




    <BarContainer object of 13 artists>




    
![png](output_20_1.png)
    


Analytic school has the most authors and fair amount of works, so it has highest rate of "popularity" in this case. German Idealism, continental and rationalism also perform good, which demonstrates again they are popular modern philosophy genre.

## Difficulty

When people want to study a school of philosophy, difficulty of that is one thing they care about. An obscure book always comprise verbose sentence and uncommon words. So here we try to measure that.

First, we create a dictionary that comprise all the words that those texts use.Raw data has a tokenized_txt. We split it up and combine them together to creat a list of all the words. This step costs several hours.


```python
token = list(map(eval, data.tokenized_txt))
total_words = sum(token,[])

# I saved the output in a csv file.
# with open('total_words_output.csv', 'w',encoding = 'utf-8') as f:
#     f.write(np.str(total_words))
```

There are 9270318 in total, which has a lot repetition of words.


```python
len(total_words) 
```




    9270318



Lets introduce FreqDist in nltk, which helps us to get the frequency of each words.The output of FreqDist is a dictionary, and in each element, it comprise the words as key and its frequency integer as value. There are 90332 words appears at least one time.


```python
from nltk.probability import FreqDist
fdist = FreqDist(total_words)

len(fdist)
```




    90332



The most common wors is "the", then by "of", which count a large proportion of whole text words. But we care about rarely appeared words, which is the main barrier to understand text. So, lets sort it by function "most_common()"


```python
fdist
```




    FreqDist({'the': 660444, 'of': 422626, 'and': 271548, 'to': 260476, 'is': 235136, 'in': 220195, 'that': 175098, 'it': 153145, 'as': 106479, 'be': 94540, ...})




```python
common_words_sort = fdist.most_common()
len(common_words_sort)
```




    90332



How to define a common appeared word? I would like to pick 99% of whole words (including repetition) as common words.


```python
num_total_words = len(total_words)
ninety_nine_percent = np.int(num_total_words*0.99)
ninety_nine_percent, num_total_words
```




    (9177614, 9270318)



Our "common_words_sort = fdist.most_common()" has been sorted form the most common words to the least common words. It's a list of tuple. In each tuple, it comprise words and number of frequency. We can add them up to find a threshold that separate common words, which is 99% common, and uncommon words, 1% left.


```python
s = 0
threshold = 0
while (s <= ninety_nine_percent):
    s = s + common_words_sort[threshold][1]
    threshold = threshold + 1
threshold, len(common_words_sort)
```




    (30000, 90332)



So, the threshold is sitting at 30000 from the most common "the". There are still over 60000 words. All of them only counts for 1% percent of total words appear. Imagine how painful when we meet thoe words.

We put the common words together in a list for reference.


```python
not_rare_words_dict = [not_rare_words[i][0] for i in range(len(not_rare_words))]
```

Define a function that take a single tokenized_txt, split it and check if each splited word is in common words dictionary. If it is not in dictionary, we can count it as a rare word (b plus one) and add it in Notions. We will apply this function together with apply() function on column of dataframe.


```python
def count_rarenotion(tokenized_txt):
    Dict=not_rare_words_dict
    b=0
    Notions=[]
    for a in tokenized_txt.strip('"["\'').strip('\']"').split("', '"):
        if a not in Dict:
            Notions.append(a)
            b+=1
    return (b,Notions)
```

Then, apply this on dataframe to create new column. data["Notions"] refers to the rare word in a sentence. data["NumOfNotions"] is the number of rare words in a sentence. This step cost a bit time.


```python
data["Notions"] = data["tokenized_txt"].apply(count_rarenotion) # output is a tuple
data["NumOfNotions"]=data["Notions"].apply(lambda x: x[0])
data["Notions"]=data["Notions"].apply(lambda x: x[1])
data["Notions"]
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>author</th>
      <th>school</th>
      <th>sentence_spacy</th>
      <th>sentence_str</th>
      <th>original_publication_date</th>
      <th>corpus_edition_date</th>
      <th>sentence_length</th>
      <th>sentence_lowered</th>
      <th>tokenized_txt</th>
      <th>lemmatized_str</th>
      <th>n_tokens</th>
      <th>Notions</th>
      <th>NumOfNotions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>What's new, Socrates, to make you leave your ...</td>
      <td>What's new, Socrates, to make you leave your ...</td>
      <td>-350</td>
      <td>1997</td>
      <td>125</td>
      <td>what's new, socrates, to make you leave your ...</td>
      <td>['what', 'new', 'socrates', 'to', 'make', 'you...</td>
      <td>what be new , Socrates , to make -PRON- lea...</td>
      <td>23</td>
      <td>[]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>Surely you are not prosecuting anyone before t...</td>
      <td>Surely you are not prosecuting anyone before t...</td>
      <td>-350</td>
      <td>1997</td>
      <td>69</td>
      <td>surely you are not prosecuting anyone before t...</td>
      <td>['surely', 'you', 'are', 'not', 'prosecuting',...</td>
      <td>surely -PRON- be not prosecute anyone before ...</td>
      <td>12</td>
      <td>[]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>The Athenians do not call this a prosecution b...</td>
      <td>The Athenians do not call this a prosecution b...</td>
      <td>-350</td>
      <td>1997</td>
      <td>74</td>
      <td>the athenians do not call this a prosecution b...</td>
      <td>['the', 'athenians', 'do', 'not', 'call', 'thi...</td>
      <td>the Athenians do not call this a prosecution ...</td>
      <td>11</td>
      <td>[]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>What is this you say?</td>
      <td>What is this you say?</td>
      <td>-350</td>
      <td>1997</td>
      <td>21</td>
      <td>what is this you say?</td>
      <td>['what', 'is', 'this', 'you', 'say']</td>
      <td>what be this -PRON- say ?</td>
      <td>5</td>
      <td>[]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Plato - Complete Works</td>
      <td>Plato</td>
      <td>plato</td>
      <td>Someone must have indicted you, for you are no...</td>
      <td>Someone must have indicted you, for you are no...</td>
      <td>-350</td>
      <td>1997</td>
      <td>101</td>
      <td>someone must have indicted you, for you are no...</td>
      <td>['someone', 'must', 'have', 'indicted', 'you',...</td>
      <td>someone must have indict -PRON- , for -PRON- ...</td>
      <td>19</td>
      <td>[]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Then we can count rare words of school and their ratio.


```python
rare_words_by_school = data.groupby("school").NumOfNotions.sum()
rare_words_by_school
```




    school
    analytic           11906
    aristotle           7473
    capitalism          2613
    communism           6036
    continental        12956
    empiricism          2737
    feminism            7801
    german_idealism    14933
    nietzsche           4644
    phenomenology       8537
    plato               5108
    rationalism         7346
    stoicism             622
    Name: NumOfNotions, dtype: int64




```python
rare_words_school_ratio = rare_words_by_school/len(rare_words)
rare_words_school_ratio = rare_words_school_ratio/sum(rare_words_school_ratio) # normalization 
rare_words_school_ratio
```




    school
    analytic           0.128419
    aristotle          0.080604
    capitalism         0.028184
    communism          0.065105
    continental        0.139745
    empiricism         0.029522
    feminism           0.084142
    german_idealism    0.161069
    nietzsche          0.050091
    phenomenology      0.092081
    plato              0.055095
    rationalism        0.079235
    stoicism           0.006709
    Name: NumOfNotions, dtype: float64



ratio is calculated by number of rare words in a school over number of total rare word. Finaly, we get chart.


```python
fig7 = plt.figure(figsize = (20,10))
plt.bar(np.sort(school), rare_words_school_ratio)
```




    <BarContainer object of 13 artists>




    
![png](output_46_1.png)
    


We can see that analytic, continental, german idealism are three top complex school with the most rare words. Modern philosophy comparing to the ancient are relativly difficult to understand.   

Now, we add another feature, the length of sentence as a part difficulty. (Long sentence often has many logic components)


```python
averge_len_school = data.groupby("school").n_tokens.mean()
averge_len_school = averge_len_school/sum(averge_len_school)
averge_len_school

fig8 = plt.figure(figsize = (20,10))
plt.bar(np.sort(school), averge_len_school)
```




    <BarContainer object of 13 artists>




    
![png](output_48_1.png)
    


There is not much diffierence between. Combine two features with equal weight as a difficulty measurement.


```python
difficulty = averge_len_school + rare_words_school_ratio

fig9 = plt.figure(figsize = (20,10))
plt.bar(np.sort(school), difficulty)
```




    <BarContainer object of 13 artists>




    
![png](output_50_1.png)
    


Morden philosophy theory such as analytic, continental and german idealism are most difficult to read under this index. Ancient philosophy such as aristotle and plato, or the philosophy people often heard like capitalsm and communism are relativly easier to understand.

We can also use this tool to analyse difficulty by author, which philosophier likes to use rare word most?


```python
rare_words_by_author = data.groupby("author").NumOfNotions.sum()
rare_words_author_ratio = rare_words_by_author/len(rare_words)
rare_words_author_ratio = rare_words_author_ratio/sum(rare_words_author_ratio) # normalization 
rare_words_author_ratio
```




    author
    Aristotle          0.080604
    Beauvoir           0.058773
    Berkeley           0.001855
    Davis              0.015467
    Deleuze            0.049206
    Derrida            0.026404
    Descartes          0.001381
    Epictetus          0.000582
    Fichte             0.006677
    Foucault           0.064134
    Hegel              0.104301
    Heidegger          0.058914
    Hume               0.009956
    Husserl            0.015403
    Kant               0.050091
    Keynes             0.006968
    Kripke             0.020019
    Leibniz            0.026523
    Lenin              0.013871
    Lewis              0.022597
    Locke              0.017711
    Malebranche        0.048850
    Marcus Aurelius    0.006126
    Marx               0.051234
    Merleau-Ponty      0.017765
    Moore              0.000874
    Nietzsche          0.050091
    Plato              0.055095
    Popper             0.006515
    Quine              0.066313
    Ricardo            0.001284
    Russell            0.003786
    Smith              0.019933
    Spinoza            0.002481
    Wittgenstein       0.008316
    Wollstonecraft     0.009902
    Name: NumOfNotions, dtype: float64




```python
fig10 = plt.figure(figsize = (35,10))
plt.bar(np.sort(au), rare_words_author_ratio)
```




    <BarContainer object of 36 artists>




    
![png](output_54_1.png)
    



```python
averge_len_author = data.groupby("author").n_tokens.mean()
averge_len_author = averge_len_author/sum(averge_len_author)
averge_len_author

fig11 = plt.figure(figsize = (32,10))
plt.bar(np.sort(au), averge_len_author)
```




    <BarContainer object of 36 artists>




    
![png](output_55_1.png)
    



```python
difficulty_by_author = averge_len_author + rare_words_author_ratio

fig12 = plt.figure(figsize = (35,10))
plt.bar(np.sort(au), difficulty_by_author)
```




    <BarContainer object of 36 artists>




    
![png](output_56_1.png)
    


Among philosophiers, Hegel's work is difficult to read, then followed by Aristotle, Foucault and Beauvoir.


```python

```
