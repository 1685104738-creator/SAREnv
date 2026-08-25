# sarenv/core/lost_person.py
"""
Generates plausible lost_person locations based on geographic features.
"""
import random
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon

from sarenv.core.loading import SARDatasetItem
from sarenv.utils.logging_setup import get_logger

log = get_logger()

class LostPersonLocationGenerator:
    """
    Generates plausible lost_person locations based on geographic features.
    """
    def __init__(self, dataset_item: SARDatasetItem, *, seed: int | None = None):
        self.dataset_item = dataset_item
        self.features = dataset_item.features.copy()
        self._python_rng = random if seed is None else random.Random(seed)
        self._pandas_random_state = (
            None if seed is None else np.random.default_rng(seed)
        )
        self.type_probabilities = {}
        self._calculate_weights()

    def _calculate_weights(self):
        if self.features.empty or 'area_probability' not in self.features.columns:
            log.warning("Features are empty or missing 'area_probability'. Cannot calculate weights.")
            return

        type_weights = self.features.groupby('feature_type')['area_probability'].sum()
        self.type_probabilities = type_weights.to_dict()
        log.info(f"Calculated feature type probabilities: {self.type_probabilities}")

    def _generate_random_point_in_polygon(self, poly: Polygon) -> Point:
        min_x, min_y, max_x, max_y = poly.bounds
        while True:
            random_point = Point(
                self._python_rng.uniform(min_x, max_x),
                self._python_rng.uniform(min_y, max_y),
            )
            if poly.contains(random_point):
                return random_point

    def generate_locations(self, n: int = 1, percent_random_samples=0) -> list[Point]:
        """
        Generate multiple plausible lost_person locations.

        Args:
            n (int): Number of locations to generate.

        Returns:
            List of shapely.geometry.Point objects (may be fewer than n if not enough valid locations).
        """
        if not self.type_probabilities:
            log.error("No feature probabilities available. Cannot generate locations.")
            return []

        locations = []

        center_proj = gpd.GeoDataFrame(
            geometry=[Point(self.dataset_item.center_point)],
            crs="EPSG:4326"
        ).to_crs(self.features.crs).geometry.iloc[0]
        main_search_circle = center_proj.buffer(self.dataset_item.radius_km * 1000)

        while len(locations) < n:
            # Randomly choose a feature type based on the probabilities
            if self._python_rng.random() < percent_random_samples:  # 10% chance to choose a random type
                chosen_feature = self.features.sample(
                    n=1,
                    random_state=self._pandas_random_state,
                ).iloc[0]
                feature_buffer = chosen_feature.geometry.buffer(15)
                final_search_area = feature_buffer.intersection(main_search_circle)
            else:
                chosen_type = self._python_rng.choices(
                    list(self.type_probabilities.keys()),
                    weights=list(self.type_probabilities.values()),
                    k=1
                )[0]
                type_gdf = self.features[self.features['feature_type'] == chosen_type]
                chosen_feature = type_gdf.sample(
                    n=1,
                    weights='area_probability',
                    random_state=self._pandas_random_state,
                ).iloc[0]
                feature_buffer = chosen_feature.geometry.buffer(15)
                final_search_area = feature_buffer.intersection(main_search_circle)


            point = self._generate_random_point_in_polygon(final_search_area)
            if point:
                locations.append(point)
        if len(locations) < n:
            log.warning(f"Only generated {len(locations)} out of {n} requested locations.")

        return locations

    # For backward compatibility
    def generate_location(self) -> Point | None:
        locations = self.generate_locations(1)
        return locations[0] if locations else None


    # def generate_location(self) -> Point | None:
    #     if not self.type_probabilities:
    #         log.error("No feature probabilities available. Cannot generate location.")
    #         return None

    #     # Create projected center point and main search circle
    #     center_proj = gpd.GeoDataFrame(geometry=[Point(self.dataset_item.center_point)], crs="EPSG:4326").to_crs(self.features.crs).geometry.iloc[0]
    #     main_search_circle = center_proj.buffer(self.dataset_item.radius_km * 1000)

    #     # Step 1: Sample a random point inside the main search circle
    #     random_point = self._generate_random_point_in_polygon(main_search_circle)
    #     if not random_point:
    #         return None

    #     # Step 2: Create a 100m buffer around that point
    #     local_buffer = gpd.GeoDataFrame(geometry=[random_point], crs=self.features.crs).buffer(100).iloc[0]

    #     # Step 3: Extract intersecting features
    #     intersecting_features = self.features[self.features.geometry.intersects(local_buffer)].copy()

    #     if intersecting_features.empty or intersecting_features['area_probability'].sum() == 0:
    #         return None

    #     # Step 4: Sample one feature based on area_probability
    #     chosen_feature = intersecting_features.sample(n=1, weights='area_probability').iloc[0]

    #     # Step 5: Intersect feature buffer with the 100m buffer and sample a final point
    #     feature_buffer = chosen_feature.geometry.buffer(10)
    #     final_search_area = feature_buffer.intersection(local_buffer)

    #     if final_search_area.is_empty:
    #         return None

    #     return self._generate_random_point_in_polygon(final_search_area)
