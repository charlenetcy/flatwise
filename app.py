from modules.preprocessing import preprocess_hdb_data
from modules.csp_filter import csp_filter_flats


def integration():
    """
    Integration with Streamlit from preprocessing to CSP filtering. 
    """
    
    # App startup
    df, _ = preprocess_hdb_data('ResaleFlatPricesData.csv', verbose=False)
    print(f"    Data loaded: {len(df)} flats in memory")
    
    # User submits form in Streamlit and pass the constraints
    user_constraints = {
        'min_price': 400000,
        'max_price': 600000,
        'towns': ['BISHAN', 'ANG MO KIO'],
        'flat_types': ['4 ROOM', '5 ROOM'],
        'min_floor_area': 90
    }
    
    filtered_df, stats = csp_filter_flats(df, user_constraints, verbose=False)
     
    print(f"!! Filtering complete: {stats}")
    
    # Filtered dataframe passed to Bayesian network, MCDA ranker, UI


if __name__ == "__main__":
    try:
        integration()
        
    except FileNotFoundError:
        print("\n!! Error: ResaleFlatPricesData.csv not found!")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n CSP filtering works!")
    