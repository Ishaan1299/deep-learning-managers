# Food Delivery Sentiment — Design Notes

## Why LSTM + Embedding (Not ANN or Bag-of-Words)

A restaurant's cuisine list is a variable-length ordered sequence of tokens. The order encodes positioning: `['North Indian', 'Chinese']` (focused, budget) differs from `['Italian', 'Mediterranean', 'Fine Dining', 'Continental']` (premium, coherent). An ANN with one-hot encoding would require 111 binary input dimensions and treat all cuisines as independent, unordered signals — losing positional context and exploding sparsity.

The Embedding layer learns a dense 32-dimensional representation per cuisine type. Cuisines that co-occur in high-rated restaurants cluster together in embedding space. The LSTM processes the sequence token-by-token; its final hidden state encodes the complete cuisine positioning profile. This is numerically equivalent to how NLP models process sentence semantics.

## Why Merge CSV and JSON (Not Use CSV Alone)

The Kaggle Zomato CSV covers all-India restaurants (8,652 rows) but is sparse for Delhi NCR. The 5 JSON API files add 29,753 Delhi NCR restaurants. Merging triples the dataset size (32,912 after filtering) and provides geographic diversity across New Delhi, Noida, Gurgaon, Guwahati, and Lucknow. The combined dataset trains a model that generalises beyond any single city.

## Why Filter "Not Rated"

5,493 combined records (14.3%) carry the label "Not rated" — the restaurant exists on Zomato but has received zero reviews. There is no rating to predict and no customer signal to learn from. Including them as a 6th class would add noise (the model cannot distinguish a new restaurant from an unreviewed old one based on features alone). They are dropped before train/test split.

## Sequence Length = 8

92% of India restaurants on Zomato offer 8 or fewer distinct cuisines. Padding to 8 tokens captures almost all real data without introducing excessive PAD tokens. The `Embedding(padding_idx=0)` setting ensures PAD tokens contribute zero gradients — they are structurally ignored during LSTM processing.

## Class Weighting

"Poor" accounts for only 2.5% of rated restaurants; "Excellent" for 6.3%. Without inverse-frequency class weights, the model defaults to predicting "Average" (33.6%) for ambiguous cases and achieves a misleadingly high overall accuracy. Class weights (Poor=8.05, Excellent=3.18) force the LSTM to treat rare rating extremes as high-priority predictions — which is exactly what matters for restaurant operators and platform managers.

## Numerical Feature Selection

Five numerical features complement the cuisine sequence:
- `price_range` (1–4): direct quality signal; premium restaurants skew Excellent
- `has_online_delivery`: associated with higher review volume and rating
- `has_table_booking`: correlates with mid-to-premium ratings
- `log_votes`: log(1+votes) reduces skew from outlier restaurants with 100K+ votes
- `log_cost`: log(1+avg_cost_for_two) reduces skew from luxury restaurants
