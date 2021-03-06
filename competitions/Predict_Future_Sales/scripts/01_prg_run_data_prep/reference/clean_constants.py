# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:30:56 2020

@author: oislen
"""
import datetime as dt

# define list of columns to group by
group_cols = ['date_block_num', 'shop_id', 'item_id']

# define aggregation dictionary to group by and aggregate
agg_dict = {'year':'first',
            'month':'first',
            'item_price':'mean',
            'item_cnt_day':'sum',
            'n_refund':'sum',
            'n_sale':'sum'
            }

# russian holidays
russian_holidays = {'2013':{'russian_new_year':(dt.date(2012, 12, 31), dt.date(2013, 1, 8)),
                            'orthodox_christmas_day':(dt.date(2013, 1, 7), dt.date(2013, 1, 7)),
                            'international_womans_day':(dt.date(2013, 3, 8), dt.date(2013, 3, 8)),
                            'labour_day':(dt.date(2013, 5, 1), dt.date(2013, 5, 1)),
                            'victory_day':(dt.date(2013, 5, 9), dt.date(2013, 5, 9)),
                            'defender_of_the_father_land_day':(dt.date(2013, 5, 10), dt.date(2013, 5, 10)),
                            'russia_day':(dt.date(2013, 6, 12), dt.date(2013, 6, 12)),
                            'unity_day':(dt.date(2013, 11, 4), dt.date(2013, 11, 4))
                            },
                    '2014':{'russian_new_year':(dt.date(2014, 1, 1), dt.date(2014, 1, 9)),
                            'orthodox_christmas_day':(dt.date(2014, 1, 7), dt.date(2014, 1, 7)),
                            'international_womans_day':(dt.date(2014, 3, 10), dt.date(2014, 3, 10)),
                            'labour_day':(dt.date(2014, 5, 1), dt.date(2014, 5, 1)),
                            'victory_day':(dt.date(2014, 5, 9), dt.date(2014, 5, 9)),
                            'russia_day':(dt.date(2014, 6, 12), dt.date(2014, 6, 12)),
                            'defender_of_the_father_land_day':(dt.date(2014, 11, 1), dt.date(2014, 11, 1)),
                            'unity_day':(dt.date(2014, 11, 4), dt.date(2014, 11, 4))
                            },
                    '2015':{'russian_new_year':(dt.date(2015, 1, 1), dt.date(2015, 1, 9)),
                            'orthodox_christmas_day':(dt.date(2015, 1, 7), dt.date(2015, 1, 7)),
                            'defender_of_the_father_land_day':(dt.date(2015, 2, 23), dt.date(2014, 2, 23)),
                            'international_womans_day':(dt.date(2015, 3, 9), dt.date(2015, 3, 9)),
                            'labour_day':(dt.date(2015, 5, 1), dt.date(2015, 5, 1)),
                            'victory_day':(dt.date(2015, 5, 11), dt.date(2015, 5, 11)),
                            'russia_day':(dt.date(2015, 6, 12), dt.date(2015, 6, 12)),
                            'unity_day':(dt.date(2015, 11, 4), dt.date(2015, 11, 4))
                            }
                    }

norm_ign_cols = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day', 'ID',
                 'data_split', 'meta_level', 'primary_key', 'holdout_subset_ind',
                 'no_sales_hist_ind', 'no_holdout_sales_hist_ind', 'item_category_id', 'item_cat_id',
                 'item_cat_sub_id', 'city_enc']