import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time

def filter_by_price(df: pd.DataFrame, 
                   min_price: Optional[float] = None,
                   max_price: Optional[float] = None) -> pd.DataFrame:
    result = df.copy()
    
    if min_price is not None:
        result = result[result['resale_price'] >= min_price]
    
    if max_price is not None:
        result = result[result['resale_price'] <= max_price]
    
    return result


def filter_by_towns(df: pd.DataFrame, 
                   towns: Optional[List[str]] = None) -> pd.DataFrame:
    result = df.copy()
    
    if towns and len(towns) > 0:
        # Converts to uppercase to handle case variations
        towns_uppercased = [town.upper() for town in towns]
        result = result[result['town'].isin(towns_uppercased)]
    
    return result


def filter_by_flat_types(df: pd.DataFrame,
                        flat_types: Optional[List[str]] = None) -> pd.DataFrame:
    result = df.copy()
    
    if flat_types and len(flat_types) > 0:
        # Converts to uppercase to handle case variations
        types_uppercased = [ft.upper() for ft in flat_types]
        result = result[result['flat_type'].isin(types_uppercased)]
    
    return result


def filter_by_floor_area(df: pd.DataFrame,
                        min_area: Optional[float] = None,
                        max_area: Optional[float] = None) -> pd.DataFrame:
    result = df.copy()
    
    if min_area is not None:
        result = result[result['floor_area_sqm'] >= min_area]
    
    if max_area is not None:
        result = result[result['floor_area_sqm'] <= max_area]
    
    return result


def filter_by_storey(df: pd.DataFrame,
                    min_storey: Optional[int] = None,
                    max_storey: Optional[int] = None) -> pd.DataFrame:
    result = df.copy()
    
    if min_storey is not None:
        result = result[result['storey_midpoint'] >= min_storey]
    
    if max_storey is not None:
        result = result[result['storey_midpoint'] <= max_storey]
    
    return result


def filter_by_remaining_lease(df: pd.DataFrame,
                             min_lease: Optional[float] = None) -> pd.DataFrame:
    result = df.copy()
    
    if min_lease is not None:
        result = result[result['remaining_lease_years'] >= min_lease]
    
    return result


def filter_by_flat_models(df: pd.DataFrame,
                         flat_models: Optional[List[str]] = None) -> pd.DataFrame:
    result = df.copy()
    
    if flat_models and len(flat_models) > 0:
        # Converts to uppercase for case-insensitive matching
        models_uppercased = [model.upper() for model in flat_models]
        result = result[result['flat_model'].isin(models_uppercased)]
    
    return result


def csp_filter_flats(df: pd.DataFrame,
                         constraints: Dict[str, Any],
                         verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filters HDB flats by applying all CSP constraints sequentially.
    
    Args:
        df: Preprocessed HDB dataframe
        constraints: Dictionary of filtering constraints
        verbose: Print filtering statistics if True
        
    Returns:
        Tuple of (filtered_dataframe, statistics_dict)
    """
    start_time = time.time()
    initial_count = len(df)
    
    if verbose:
        print(f"Starting with {initial_count} flats")
    
    df_filtered = df.copy()
    
    # 1. Price constraint
    if 'min_price' in constraints or 'max_price' in constraints:
        length = len(df_filtered)
        df_filtered = filter_by_price(
            df_filtered,
            constraints.get('min_price'),
            constraints.get('max_price')
        )
        if verbose:
            print(f"After price filter: {len(df_filtered)} flats (removed {length - len(df_filtered)})")
    
    # 2. Town constraint
    if 'towns' in constraints:
        length = len(df_filtered)
        df_filtered = filter_by_towns(df_filtered, constraints['towns'])
        if verbose:
            print(f"After town filter: {len(df_filtered)} flats (removed {length - len(df_filtered)})")
    
    # 3. Flat type constraint
    if 'flat_types' in constraints:
        length = len(df_filtered)
        df_filtered = filter_by_flat_types(df_filtered, constraints['flat_types'])
        if verbose:
            print(f"After flat type filter: {len(df_filtered)} flats (removed {length - len(df_filtered)})")
    
    # 4. Floor area constraint
    if 'min_floor_area' in constraints or 'max_floor_area' in constraints:
        length = len(df_filtered)
        df_filtered = filter_by_floor_area(
            df_filtered,
            constraints.get('min_floor_area'),
            constraints.get('max_floor_area')
        )
        if verbose:
            print(f"After floor area filter: {len(df_filtered)} flats (removed {length - len(df_filtered)})")
    
    # 5. Storey constraint
    if 'min_storey' in constraints or 'max_storey' in constraints:
        length = len(df_filtered)
        df_filtered = filter_by_storey(
            df_filtered,
            constraints.get('min_storey'),
            constraints.get('max_storey')
        )
        if verbose:
            print(f"After storey filter: {len(df_filtered)} flats (removed {length - len(df_filtered)})")
    
    # 6. Remaining lease constraint
    if 'min_remaining_lease' in constraints:
        length = len(df_filtered)
        df_filtered = filter_by_remaining_lease(
            df_filtered,
            constraints['min_remaining_lease']
        )
        if verbose:
            print(f"After lease filter: {len(df_filtered)} flats (removed {length - len(df_filtered)})")
    
    # 7. Flat model constraint
    if 'flat_models' in constraints:
        length = len(df_filtered)
        df_filtered = filter_by_flat_models(df_filtered, constraints['flat_models'])
        if verbose:
            print(f"After flat model filter: {len(df_filtered)} flats (removed {length - len(df_filtered)})")
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    stats = get_filter_statistics(df, df_filtered)
    
    if verbose:
        print(f"Final result: {len(df_filtered)} flats")
        print(f"Filtered out: {initial_count - len(df_filtered)} flats ({100 * (initial_count - len(df_filtered)) / initial_count:.1f}%)")
        print(f"Time taken: {elapsed_time:.4f} seconds")
    
    return df_filtered, stats


def get_filter_statistics(original_df: pd.DataFrame, 
                         filtered_df: pd.DataFrame) -> Dict[str, Any]:
    original_size = len(original_df)
    
    stats = {
        'total_results': len(filtered_df),
        'percentage_of_original': 100 * len(filtered_df) / original_size if original_size > 0 else 0,
        'price_range': {
            'min': filtered_df['resale_price'].min() if len(filtered_df) > 0 else None,
            'max': filtered_df['resale_price'].max() if len(filtered_df) > 0 else None,
            'median': filtered_df['resale_price'].median() if len(filtered_df) > 0 else None
        },
        'towns_present': sorted(filtered_df['town'].unique().tolist()) if len(filtered_df) > 0 else [],
        'flat_types_present': sorted(filtered_df['flat_type'].unique().tolist()) if len(filtered_df) > 0 else []
    }
    return stats



if __name__ == "__main__":
    test_data = pd.DataFrame({
        'town': ['BISHAN', 'ANG MO KIO', 'TAMPINES', 'BISHAN', 'QUEENSTOWN'],
        'flat_type': ['4 ROOM', '5 ROOM', '3 ROOM', '4 ROOM', '5 ROOM'],
        'resale_price': [450000, 550000, 300000, 480000, 600000],
        'floor_area_sqm': [95, 110, 75, 92, 120],
        'storey_midpoint': [8, 12, 5, 10, 15],
        'remaining_lease_years': [65, 70, 80, 60, 55],
        'flat_model': ['Model A', 'Improved', 'Standard', 'Model A', 'Premium']
    })
    print(test_data)
    
    test_constraints = {
        'min_price': 400000,
        'max_price': 550000,
        'towns': ['BISHAN', 'ANG MO KIO'],
        'flat_types': ['4 ROOM', '5 ROOM'],
        'min_floor_area': 90
    }
   
    resultdf, stats = csp_filter_flats(test_data, test_constraints, verbose=True)
    
    print("\n\nFiltered Results:")
    print(resultdf)
    
    print(f"Total results: {stats['total_results']}")
    print(f"Percentage of original: {stats['percentage_of_original']:.1f}%")
    print(f"Price range: ${stats['price_range']['min']} - ${stats['price_range']['max']}")
    