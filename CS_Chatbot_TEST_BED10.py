# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
import calendar
import datetime as dt
from datetime import datetime
from datetime import date
from datetime import timedelta
import re
from fuzzywuzzy import fuzz, process
import streamlit as st

import os

## CHANGING THE BASE DIRECTORY PATH TO WHERE THE DATA RESIDES 
os.chdir(r'C:\Users\Admin\chatbot_on_apparels\TABLES_MASTER')

date_parser = lambda x: pd.to_datetime(x).date()

## IMPORTING THE ORDERS DATA
orders_df = pd.read_csv('orders_data.csv',parse_dates = ['expected_date_of_delivery',
                                                         'expected_date_of_refund',
                                                         'expected_date_of_return_pickup',
                                                          'expected_date_of_replacement',
                                                          'updated_status_date'],
                                                          date_parser = date_parser)
orders_df['order_id'] = orders_df['order_id'].astype(str)
#orders_df

## IMPORTING THE PRODUCTS DATA
products_df = pd.read_csv('products_data.csv')

# DOWNLOAD NECESSARY NLTK DATA
nltk.download('punkt')
nltk.download('wordnet')

# LOAD THE INTENTS FILE AND CALLING THE LEMMATIZER FUNCTION THE DATA
with open('intents_consol.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# PREPROCESSING OF DATA
corpus = []
labels = []
classes = []
responses = {}



## UNDERSTANDING THE PATTERNS MENTIONED IN THE INTENTS, CONVERTING THE SAME INTO TOKENS (TOKENIZING) AND THEN LEMMATIZING THE DATA
## TAGGING THE RESPONSES BASED ON INTENTS INTERPRETATION FOR MAKING THE MODEL LEARN
for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        corpus.append(' '.join([lemmatizer.lemmatize(token.lower()) for token in tokens]))
        labels.append(intent['tag'])
    
    responses[intent['tag']] = intent['responses']
    classes.append(intent['tag'])

## USE OF Tfidf VECTORIZATION ALGORITHM ON PATTERNS AND MATCHING THEM TO LABELS
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(corpus)
y = np.array(labels)

## TRAIN THE MODEL ON LINEAR  SUPPORT VECTOR CLASSIFIER
model = LinearSVC()
model.fit(X, y)

## FUNCTION TO GET THE ORDERS AGAINST THE USERID
def userid_orders(userid):
    distinct_orders = orders_df[(orders_df['user_id']==userid)]['order_id'].unique()
    comma_separated_distinct_orders = ', '.join(distinct_orders)
    return comma_separated_distinct_orders

## FUNCTION TO VALIDATE THE ORDERID AS ENTERERED BY USER
def validate_order_id(orderid):
    orderid = str(orderid)
    return orderid in orders_df[(orders_df['user_id'] == userid)]['order_id'].values

def get_order_details(orderid):
    return orders_df.loc[(orders_df['user_id'] == userid) & (orders_df['order_id'] == orderid)].iloc[0]


def get_user_tag(userid):
    user_orders = orders_df[orders_df['user_id'] == userid]
    
    order_count = user_orders[user_orders['status'] == 'Order Delivery'].shape[0]
    
    avg_order_value = user_orders[user_orders['status'] == 'Order Delivery']['order_value'].mean()
    
    if order_count <=1:
        tag = "BS"
    elif order_count>=2:
        if avg_order_value<=1500:
            tag = "L"
        elif avg_order_value > 1500 and avg_order_value<=4000:
            tag = "M"
        else:
            tag = "H"
    return tag

def is_valid_date(date_string): 
    try: 
        datetime.strptime(date_string, '%Y-%m-%d'),"Socks","Neck Tie" 
        return True 
    except ValueError: 
        return False
    

    
##################################################################################


## LIST OF KEYWORDS 
keywords = [
    "Casual Shoes", "Tshirts", "Jackets", "Boxers", "Sunglasses", "Shirts", 
    "Belts", "Shorts", "Watches", "Jeans", "Ethnic Wear", "Hoodies", 
    "Sandals", "Formal Shoes", "Sweaters","Trousers","Perfumes","Deodorants","Suits","Blazer","Socks","Neck Tie"
]

## JOIN THE KEYWORDS INTO A SINGLE REGEX PATTERN
keywords_pattern = re.compile(r'|'.join([re.escape(keyword) for keyword in keywords]), re.IGNORECASE)


## REGULAR EXPRESSIONS FOR DIFFERENT PATTERNS
greater_than_pattern = re.compile(r'(?:>\s?\d+|gt\s?\d+|grt\s?\d+|grt\s?than\s?\d+|>\s?than\s?\d+|GREATER\s?THAN\s?\d+)', re.IGNORECASE)
less_than_pattern = re.compile(r'(?:<\s?\d+|lt\s?\d+|LT\s?\d+|lt\s?than\s?\d+|<\s?than\s?\d+|lESS\s?THAN\s?\d+)', re.IGNORECASE)
range_pattern = re.compile(r'(?:>\s?\d+\s?and\s?<\s?\d+|grt\s?\d+\s?lt\s?\d+|gt\s?\d+\s?lt\s?\d+|greater\s?than\s?\d+\s?AND\s?less\s?than\s?\d+|>\s?than\s?\d+\s?AND\s?<\s?than\s?\d+|between\s?\d+\s?and\s?\d+|BTW\s?\d+\s?AND\s?\d+|between\s?\d+\s?&\s?\d+|btw\s?\d+\s?&\s?\d+)', re.IGNORECASE)

## FUNCTION TO EXTRACT VALUES AND KEYWORDS
def extract_min_max_category(text_input):
    matches = re.findall(r'\d+', text_input)
    keywords_found = re.findall(keywords_pattern, text_input)
    
    grt_than = less_than = None
    if re.search(range_pattern, text_input):
        if len(matches) == 2:
            grt_than, less_than = int(matches[0]), int(matches[1])
    elif re.search(greater_than_pattern, text_input):
        if len(matches) == 1:
            grt_than = int(matches[0])
    elif re.search(less_than_pattern, text_input):
        if len(matches) == 1:
            less_than = int(matches[0])
    
    ## FUZZY MATCHING FOR KEYWORDS
    keywords_found = []
    for keyword in keywords:
        if process.extractOne(keyword, text_input.split(), scorer=fuzz.ratio)[1] >= 80:
            keywords_found.append(keyword)
    product_category = keywords_found[0] if keywords_found else 'all'
    
    
    grt_than = 0 if grt_than is None else grt_than
    less_than = 0 if less_than is None else less_than
    return grt_than, less_than, product_category

## 
def product_price_range_details(max_val,min_val,category_value):
    if max_val > 0 and min_val == 0 and category_value == 'all':
        product_price_range_df = products_df.loc[products_df['price'] > max_val, ['product_id', 'product_name', 'price']]
        records_count = product_price_range_df.shape[0]
    elif max_val == 0 and min_val > 0 and category_value == 'all':
        product_price_range_df = products_df.loc[products_df['price'] < min_val, ['product_id', 'product_name', 'price']]
        records_count = product_price_range_df.shape[0]
    elif max_val > 0 and min_val > 0 and category_value == 'all':
        product_price_range_df = products_df.loc[(products_df['price'] > max_val) & (products_df['price'] < min_val), ['product_id', 'product_name', 'price']]
        records_count = product_price_range_df.shape[0]
    elif max_val > 0 and min_val == 0 and category_value != 'all':
        product_price_range_df = products_df.loc[(products_df['price'] > max_val) & (products_df['category'].str.lower() == category_value.lower()), ['product_id', 'product_name', 'price']]
        records_count = product_price_range_df.shape[0]
    elif max_val == 0 and min_val > 0 and category_value != 'all':
        product_price_range_df = products_df.loc[(products_df['price'] < min_val) & (products_df['category'].str.lower() == category_value.lower()), ['product_id', 'product_name', 'price']]
        records_count = product_price_range_df.shape[0]
    elif max_val > 0 and min_val > 0 and category_value != 'all':
        product_price_range_df = products_df.loc[(products_df['price'] > max_val) & (products_df['price'] < min_val) & (products_df['category'].str.lower() == category_value.lower()), ['product_id', 'product_name', 'price']]
        records_count = product_price_range_df.shape[0]
    else:
        print (f"not able to get the desired data as per the inputs")
    return product_price_range_df,category_value,records_count
        
def product_details():
    distinct_products = products_df['category'].unique()
    comma_separated_distinct_products = ', '.join(distinct_products)
    return comma_separated_distinct_products

def product_wise_details(category,tag):
    if tag == "BS":
        product_category_df1 = products_df[(products_df['category'].str.lower() == category)& (products_df['new_tag'] == tag)]
    else:
        product_category_df1 = products_df[(products_df['category'].str.lower() == category)& (products_df['tag'] == tag)]
    
    product_category_df = product_category_df1[['product_id', 'product_name', 'details']]
    product_category_recommend_df = product_category_df1[['product_id', 'product_name', 'details',
                                                         'r_product_id', 'r_product_name', 'r_details']]
    
    product_category_df = product_category_df.head()
    product_category_recommend_df = product_category_recommend_df.head()
    return product_category_df,product_category_recommend_df


def price_details():
    min_price = products_df['price'].min()
    max_price = products_df['price'].max()
    return min_price,max_price

def product_price_wise_details(category,tag):
    if tag == "BS":
        product_category_price_df1 = products_df[(products_df['category'].str.lower() == category) & (products_df['new_tag'] == tag)]
    else:
        product_category_price_df1 = products_df[(products_df['category'].str.lower() == category) & (products_df['tag'] == tag)]
    
    product_category_price_df = product_category_price_df1[['product_id', 'product_name', 'price','net_discounted_price']]
    product_category_price_recommend_df = product_category_price_df1[['product_id', 'product_name', 'price','net_discounted_price',
                                                                      'r_product_id', 'r_product_name', 'r_price','r_net_discounted_price']]
    
    product_category_price = product_category_price_df.head()
    product_category_recommend_price = product_category_price_recommend_df.head()
    
    min_category_price = product_category_price_df['net_discounted_price'].min()
    max_category_price = product_category_price_df['net_discounted_price'].max()
    return product_category_price,product_category_recommend_price,min_category_price,max_category_price


def offer_details():
    distinct_offers = products_df[(products_df['offer']!= 'No discount')]['offer'].unique()
    comma_separated_distinct_offers = ', '.join(distinct_offers)
    return comma_separated_distinct_offers



def product_offer_wise_details(category,tag):
    if tag == "BS":
        product_category_offer_price_df1 = products_df[(products_df['category'].str.lower() == category) & (products_df['offer']!= 'No discount') & (products_df['new_tag'] == tag)]
    else:
        product_category_offer_price_df1 = products_df[(products_df['category'].str.lower() == category) & (products_df['offer']!= 'No discount') & (products_df['tag'] == tag)]
    
    product_category_offer_price_df = product_category_offer_price_df1[['product_id', 'product_name', 'price','offer']]
    product_category_offer_recommend_price_df = product_category_offer_price_df1[['product_id', 'product_name', 'price','offer',
                                                                                  'r_product_id', 'r_product_name', 'r_price','r_offer']]
    
    product_category_offer_price = product_category_offer_price_df.head()
    product_category_offer_recommend_price = product_category_offer_recommend_price_df.head()
    
    distinct_category_offers = product_category_offer_price_df[(product_category_offer_price_df['offer']!= 'No discount')]['offer'].unique()
    comma_separated_distinct_category_offers = ', '.join(distinct_category_offers)
    return product_category_offer_price,product_category_offer_recommend_price,comma_separated_distinct_category_offers




def chatbot_response(text):
    tokens = nltk.word_tokenize(text)
    input_vec = vectorizer.transform([' '.join([lemmatizer.lemmatize(token.lower()) for token in tokens])])
    prediction = model.predict(input_vec)[0]
    
    
    if prediction == 'product_details':
        product_category_master = product_details()
        return f"We sell products in the following categories :'{product_category_master}'."
    
    if prediction == 'price_details':
        min_products_price,max_products_price = price_details()
        return f"We sell products in men's clothing,footwear & accessories categories in the price range of Rs {min_products_price} to Rs {max_products_price}."
        
     
    if prediction == 'offer' or prediction == 'discount':
        offer_master = offer_details()
        return f"The current offers/discounts available are : {offer_master}."
    
    if prediction in ['casual_shoes_details','tshirts_details','jackets_details','boxers_details','sunglasses_details',
                      'shirts_details','belts_details','shorts_details','watches_details','jeans_details',
                      'ethnic_wear_details','hoodies_details','sandals_details','formal_shoes_details','sweaters_details',
                      'trousers_details','perfumes_details','deodarants_details','suits_details','blazers_details','socks_details',
                      'neck_tie_details'
    ]:
        product_category_split = prediction.split('_')
        product_category_actual = ' '.join(product_category_split[:-1])
        user_tag = get_user_tag(userid)
        
        product_category_specific_df,product_category_specific_recommend_df = product_wise_details(product_category_actual,user_tag)
        return (f"below are the details of best recommended products specially for you in {product_category_actual} category", 
                product_category_specific_df,
                f"\nAdditionally,we recommend the following best combinations for {product_category_actual} which can be bundled together",
                product_category_specific_recommend_df,
                f"For more details visit the website : www.gentsgear.com/{product_category_actual.replace(' ','-')}")
                
               
 
    if prediction in ['casual_shoes_price','tshirts_price','jackets_price','boxers_price','sunglasses_price',
                      'shirts_price','belts_price','shorts_price','watches_price','jeans_price',
                      'ethnic_wear_price','hoodies_price','sandals_price','formal_shoes_price','sweaters_price',
                      'trousers_price','perfumes_price','deodarants_price','suits_price','blazers_price','socks_price',
                      'neck_tie_price'
    ]:
        product_category_price_split = prediction.split('_')
        product_category_price_actual = ' '.join(product_category_price_split[:-1])
        user_tag = get_user_tag(userid)
        product_category_price_specific_df,product_category_price_specific_recommend_df,min_category_price,max_category_price = product_price_wise_details(product_category_price_actual,user_tag)
        return (f"below are the details of the best recommended products specially for you in the {product_category_price_actual} category", 
                product_category_price_specific_df,
                f"{product_category_price_actual} in the price range of Rs {min_category_price} to Rs {max_category_price} are available.",
                f"\nAdditionally,we recommend the following best combinations for {product_category_price_actual} which can be bundled together",
                product_category_price_specific_recommend_df,            
                f"For more details visit the website : www.gentsgear.com/{product_category_price_actual.replace(' ','-')}")
        
    
    if prediction in ['casual_shoes_offer','tshirts_offer','jackets_offer','boxers_offer','sunglasses_offer',
                      'shirts_offer','belts_offer','shorts_offer','watches_offer','jeans_offer',
                      'ethnic_wear_offer','hoodies_offer','sandals_offer','formal_shoes_offer','sweaters_offer',
                      'trousers_offer','perfumes_offer','deodarants_offer','suits_offer','blazers_offer','socks_offer',
                      'neck_tie_offer'   
    ]:
        product_category_offer_split = prediction.split('_')
        product_category_offer_actual = ' '.join(product_category_offer_split[:-1])
        user_tag = get_user_tag(userid)
        product_category_offer_specific_df,product_category_offer_recommend_specific_df,category_specific_offers = product_offer_wise_details(product_category_offer_actual,user_tag)
        return (f"below are the details of the best recommended products specially for you with offers in the {product_category_offer_actual} category:",
                product_category_offer_specific_df,
                f"The following offers are there in {product_category_offer_actual} "
                f"category :\n {category_specific_offers}",
                f"\nAdditionally,we recommend the following best combinations for {product_category_offer_actual} which can be bundled together",
                product_category_offer_recommend_specific_df,
                f"For more details visit the website : www.gentsgear.com/{product_category_offer_actual.replace(' ','-')}")
    
    if prediction in ['price_range']:
        max_value,min_value,category = extract_min_max_category(text)
        product_price_range_df_data,cat_value,row_count = product_price_range_details(max_value,min_value,category)
        return (f"below are the details of the products in the {cat_value} category:",
                product_price_range_df_data,
                f"There are {row_count} {cat_value} found in your chosen price range")
    
    if prediction in ['complaint_existing']:
        st.session_state.awaiting_new_existing_complaint = True
        return ("Please type 'Y' if this is for an existing booking, otherwise type 'N'."
            "\nType 'exit' to go to the main menu.")

    if st.session_state.get('awaiting_new_existing_complaint'):
        st.session_state.input = text
        input = st.session_state.input
        if input.lower() == 'y':
            unique_orders_for_userid = userid_orders(userid)
            st.session_state.awaiting_new_existing_complaint = False
            st.session_state.awaiting_order_id_complaint = True
            return (f"Dear {userid}, We have found the following orders: {unique_orders_for_userid} in our system in the last 3 months."
                "\nPlease enter the order ID for which you want to register a complaint or type 'exit' to go to the main menu.")
        elif input.lower() == 'n':
            st.session_state.awaiting_new_existing_complaint = False
            st.session_state.awaiting_new_complaint_reason = True
            return (f"Dear {userid}, Please describe the issue or type 'exit' to go to the main menu.")
        else:
            st.session_state.awaiting_new_existing_complaint = False
            return "Returning to the main menu."

    if st.session_state.get('awaiting_order_id_complaint'):
        if text.lower() == 'exit':
            st.session_state.awaiting_order_id_complaint = False
            return "Returning to the main menu."
        if not validate_order_id(text):
            return "Invalid order ID. Please enter a valid order ID or type 'exit' to go to the main menu."

        st.session_state.order_id = text
        order_details = get_order_details(text)
        st.session_state.order_details = order_details
        st.session_state.awaiting_order_id_complaint = False
        st.session_state.awaiting_existing_complaint_reason = True
        return "Please describe the issue in detail or type 'exit' to go to the main menu:"

    if st.session_state.get('awaiting_existing_complaint_reason'):
        if text.lower() == 'exit':
            st.session_state.awaiting_existing_complaint_reason = False
            return "Returning to the main menu."

        update_reason = text
        orderid = st.session_state.order_id
        orders_df.loc[orders_df['order_id'] == orderid, 'updated_status'] = 'order complaint'
        orders_df.loc[orders_df['order_id'] == orderid, 'updated_status_date'] = pd.to_datetime(datetime.today()).date()
        orders_df.loc[orders_df['order_id'] == orderid, 'updated_status_reason'] = update_reason
        orders_df.to_csv('orders_data.csv', index=False)
        st.session_state.order_details['updated_status'] = 'order complaint'
        st.session_state.order_details['updated_status_date'] = pd.to_datetime(datetime.today()).date()
        st.session_state.order_details['updated_status_reason'] = update_reason
        st.session_state.awaiting_existing_complaint_reason = False
        return ("Your complaint has been registered.\n"
            "Thank you for your input. Our customer care executive will call you back within 2 hours.")

    if st.session_state.get('awaiting_new_complaint_reason'):
        if text.lower() == 'exit':
            st.session_state.awaiting_new_complaint_reason = False
            return "Returning to the main menu."

        st.session_state.complaint = text
        st.session_state.awaiting_new_complaint_reason = False
        return ("Your complaint has been registered.\n"
                "Thank you for your input. Our customer care executive will call you back within 2 hours.")

    
    elif prediction in ['order_status', 'expected_delivery_time', 'expected_date_of_delivery',
                    'expected_date_of_refund', 'expected_date_of_return', 'expected_date_of_replacement',
                    'order_cancellation', 'order_rescheduling', 'other_request']:

        st.session_state.awaiting_order_id = True
        st.session_state.prediction = prediction
        unique_orders_for_userid = userid_orders(userid)
        return (f"Dear {userid}, We have found the following orders :{unique_orders_for_userid} in our system in the last 3 month\n",
                f"Please enter the order ID or type 'exit' to go to the main menu.")

    if st.session_state.get('awaiting_order_id'):
        if text.lower() == 'exit':
            st.session_state.awaiting_order_id = False
            return "Returning to the main menu."
        if not validate_order_id(text):
            return "Invalid order ID. Please enter a valid order ID or type 'exit' to go to the main menu."

        st.session_state.order_id = text
        st.session_state.awaiting_order_id = False
        order_details = get_order_details(text)
        st.session_state.order_details = order_details
    
        prediction = st.session_state.prediction

        # Handle different cases after order ID validation
        if prediction == 'order_status':
            return f"Your order status is '{order_details['status']}'."

        elif prediction == 'expected_delivery_time':
            return f"Your expected delivery date is '{order_details['expected_date_of_delivery']}'."

        elif prediction == 'expected_date_of_delivery':
            return f"Your expected delivery date is '{order_details['expected_date_of_delivery']}'."

        elif prediction == 'expected_date_of_refund':
            return f"Your expected date of refund is '{order_details['expected_date_of_refund']}'."

        elif prediction == 'expected_date_of_return':
            return f"Your expected date of return pickup is '{order_details['expected_date_of_return_pickup']}'."

        elif prediction == 'expected_date_of_replacement':
            return f"Your expected date of replacement delivery is '{order_details['expected_date_of_replacement']}'."

        elif prediction == 'order_cancellation':
            if order_details['status'] in ['Order Placement', 'Order Confirmation'] and order_details['updated_status'] not in ['order cancelled', 'order rescheduled']:
                st.session_state.awaiting_reason = True
                return "Please provide reason for cancellation:"
            else:
                return f"Your order can't be cancelled as it is already in {order_details['updated_status'] if order_details['updated_status'] not in ['', 'na'] else order_details['status']} stage."
            
            
            
        elif prediction == 'order_rescheduling':
            if order_details['status'] in ['Order Placement', 'Order Confirmation'] and order_details['updated_status'] not in ['order cancelled', 'order rescheduled']:
                st.session_state.awaiting_reschedule_date = True
                return "Please provide the new date for rescheduling (YYYY-MM-DD) or type 'exit' to go to the main menu:"
            else:
                return f"Your order can't be rescheduled as it is already in {order_details['updated_status'] if order_details['updated_status'] not in ['', 'na'] else order_details['status']} stage."
        

        
    if st.session_state.get('awaiting_reason'):
        update_reason = text
        orderid = st.session_state.order_id
        orders_df.loc[orders_df['order_id'] == orderid, 'updated_status'] = 'order cancelled'
        orders_df.loc[orders_df['order_id'] == orderid, 'updated_status_date'] = pd.to_datetime(datetime.today()).date()
        orders_df.loc[orders_df['order_id'] == orderid, 'updated_status_reason'] = update_reason
        orders_df.to_csv('orders_data.csv',index = False)
        st.session_state.order_details['updated_status'] = 'order cancelled'
        st.session_state.order_details['updated_status_date'] = pd.to_datetime(datetime.today()).date()
        st.session_state.order_details['updated_status_reason'] = update_reason
        st.session_state.awaiting_reason = False
        return "Your order has been cancelled and the refund will be processed within 5-7 business days."

    if st.session_state.get('awaiting_reschedule_date'):
        new_date = text
        if new_date.lower() == 'exit':
            st.session_state.awaiting_reschedule_date = False
            return "Returning to the main menu."
        if not is_valid_date(new_date):
            return "Invalid date format. Please enter the date in YYYY-MM-DD format."

        reschedule_date = pd.to_datetime(new_date)
        st.session_state.reschedule_date = reschedule_date
        day_diff = (reschedule_date - st.session_state.order_details['expected_date_of_delivery']).days
        if day_diff >0 and day_diff <= 3:
            #st.session_state.reschedule_date = reschedule_date
            st.session_state.awaiting_reschedule_date = False
            st.session_state.awaiting_reason_reschedule = True
            return "Please provide reason for rescheduling the order:"
        else:
            return "The reschedule date can't be more than 3 days from the expected delivery date."

    
    if st.session_state.get('awaiting_reason_reschedule'):
        update_reason = text
        orderid = st.session_state.order_id
        reschedule_date = st.session_state.reschedule_date
        orders_df.loc[orders_df['order_id'] == orderid, 'updated_status'] = 'order rescheduled'
        orders_df.loc[orders_df['order_id'] == orderid, 'updated_status_date'] = reschedule_date
        orders_df.loc[orders_df['order_id'] == orderid, 'updated_status_reason'] = update_reason
        orders_df.to_csv('orders_data.csv',index = False)
        st.session_state.order_details['updated_status'] = 'order rescheduled'
        st.session_state.order_details['updated_status_date'] = reschedule_date
        st.session_state.order_details['updated_status_reason'] = update_reason
        st.session_state.awaiting_reason_reschedule = False
        return f"Your order has been rescheduled to {reschedule_date}."

    elif prediction == 'other_request':
        return "Please provide details of your request so we can assist you further."

    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"
 
        
    
    


# Chat with the bot
st.set_page_config(
    page_title="Chat App",
    layout="wide",
    initial_sidebar_state="collapsed",
)

userid = "Nitesh"
st.title(f"Hi {userid}!! How can I assist you today?")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], tuple):  # Check if the response is a tuple
            for element in msg["content"]:
                if isinstance(element, pd.DataFrame):  # Check if the element is a DataFrame
                    st.dataframe(element)
                else:
                    st.write(element)  # Display text or other types
        else:
            st.write(msg["content"])  # Display regular text messages

prompt = st.chat_input("Say something...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Call the chatbot response function
    bot_response = chatbot_response(prompt)
    
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    with st.chat_message("bot"):
        if isinstance(bot_response, tuple):  # Check if the response is a tuple
            for element in bot_response:
                if isinstance(element, pd.DataFrame):  # Check if the element is a DataFrame
                    st.dataframe(element)
                else:
                    st.write(element)  # Display text or other types
        else:
            st.write(bot_response)  # Display regular text messages
	
# -


