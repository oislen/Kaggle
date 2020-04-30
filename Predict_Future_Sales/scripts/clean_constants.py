# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:30:56 2020

@author: oislen
"""

# define list of columns to group by
group_cols = ['year', 'month', 'shop_id', 'item_id']

# define aggregation dictionary to group by and aggregate
agg_dict = {'date_block_num':'first',
            'item_price':'mean',
            'item_cnt_day':'sum',
            'n_refund':'sum',
            'n_sale':'sum',
            'price_decimal':'mean',
            'price_decimal_len':'mean',
            'item_name':'first',
            'item_category_id':'first',
            'item_category_name':'first',
            'item_cat':'first',
            'item_cat_sub':'first',
            'shop_name':'first',
            'shop_quotes':'first',
            'shop_brackets':'first',
            'shop_smooth':'first'
            }