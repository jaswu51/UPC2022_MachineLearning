The Process to reproduce our result:

1. Downlod the kaggle file with 
   ```
   kaggle kernels output hervind/h-m-faster-trending-products-weekly -p /path/to/dest
   ```
2. Install model and requirements in https://github.com/openai/CLIP. Change the path to Kaggle dataset article.csv and run 
   ```
   python clip_based_feature_converter.py
   ```
3. Run FeatureEngineering.ipynb to do feature engineering and PCA, create atomic file(File format for Recbole).
   
4. Run follows to compare the validation and test result of different model
   ```
   python run_compare_model.py
   ```
   
5. Create initial submission and replace by LightSans result.
   ```
   python base_submission.py
   python run_LightSANs.py
   ```