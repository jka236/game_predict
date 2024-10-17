import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class FeatureEngineer:
    def __init__(self, players_df, games_df):
        self.players_df = players_df
        self.games_df = games_df
        self.features = []
        self.feature_indices = {}  # To keep track of feature indices

    def add_feature(self, feature):
        if not isinstance(feature, Feature):
            raise TypeError("Feature must be an instance of Feature class")
        start_index = sum(len(f.get_feature_names()) for f in self.features)
        self.feature_indices[type(feature).__name__] = (start_index, start_index + len(feature.get_feature_names()))
        self.features.append(feature)

    def create_features(self, df, is_test=False):
        feature_vector = []
        for feature in self.features:
            feature_vector.extend(feature.compute(df, is_test))
        return feature_vector

    def get_feature_names(self):
        return [name for feature in self.features for name in feature.get_feature_names()]

    def remove_feature_from_dataframe(self, df, feature_class):
        if feature_class.__name__ not in self.feature_indices:
            raise ValueError(f"Feature {feature_class.__name__} not found")
        
        start, end = self.feature_indices[feature_class.__name__]
        df['features'] = df['features'].apply(lambda x: np.delete(x, range(start, end)))
        
        # Update indices for remaining features
        for name, (s, e) in self.feature_indices.items():
            if s > end:
                self.feature_indices[name] = (s - (end - start), e - (end - start))
        
        del self.feature_indices[feature_class.__name__]
        self.features = [f for f in self.features if not isinstance(f, feature_class)]

class Feature(ABC):
    @abstractmethod
    def compute(self, row, is_test, index = None):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass

class PlayerStatsFeature(Feature):
    def __init__(self, players_df, stats):
        self.players_df = players_df
        self.stats = stats

    def compute(self, row, is_test):
        year = 2015 if is_test else row['year']
        features = []
        for i in range(1, 12):
            for team in ['home', 'away']:
                player_id = row[f'{team}_player_{i}']
                if pd.notna(player_id):
                    player_data = self.players_df[(self.players_df['player_id'] == player_id) & 
                                                  (self.players_df['year'] <= year)].sort_values('year', ascending=False).iloc[0]
                    features.extend(player_data[self.stats].values)
                else:
                    features.extend([0] * len(self.stats))
        return features

    def get_feature_names(self):
        return [f"{team}_player_{i}_{stat}" for team in ['home', 'away'] for i in range(1, 12) for stat in self.stats]


class TeamHistoricalGoalsFeature(Feature):
    def __init__(self, games_df):
        self.games_df = games_df

    def compute(self, row, is_test):
        year = 2015 if is_test else row['year']
        features = []
        for team in ['home', 'away']:
            team_id = row[f'{team}_team_id']
            team_games = self.games_df[(self.games_df['home_team_id'] == team_id) | 
                                       (self.games_df['away_team_id'] == team_id) & 
                                       (self.games_df['year'] <= year)]
            home_goals = team_games[team_games['home_team_id'] == team_id]['home_team_goal'].mean()
            away_goals = team_games[team_games['away_team_id'] == team_id]['away_team_goal'].mean()
            total_goals = (team_games[team_games['home_team_id'] == team_id]['home_team_goal'].sum() + 
                           team_games[team_games['away_team_id'] == team_id]['away_team_goal'].sum()) / len(team_games)
            features.extend([home_goals, away_goals, total_goals])
        return features

    def get_feature_names(self):
        return ['home_team_home_goal_mean', 'home_team_away_goal_mean', 'home_team_total_goal_mean',
                'away_team_home_goal_mean', 'away_team_away_goal_mean', 'away_team_total_goal_mean']


class PositionRatingsFeature(Feature):
    def __init__(self, players_df):
        self.players_df = players_df
        self.positions = ['GK', 'DEF', 'MID', 'FWD']

    def compute(self, row, is_test):
        year = 2015 if is_test else row['year']
        features = []
        for team in ['home', 'away']:
            ratings = {pos: [] for pos in self.positions}
            for i in range(1, 12):
                player_id = row[f'{team}_player_{i}']
                if pd.notna(player_id):
                    player_data = self.players_df[(self.players_df['player_id'] == player_id) & 
                                                  (self.players_df['year'] <= year)].sort_values('year', ascending=False).iloc[0]
                    position = self.positions[min(int(row[f'{team}_player_Y{i}']) // 3, 3)]
                    ratings[position].append(player_data['overall_rating'])
            features.extend([np.mean(r) if r else 0 for r in ratings.values()])
        return features
    
    def get_feature_names(self):
        return ["home_GK_rating", "away_GK_rating", "home_DEF_rating", "away_DEF_rating", "home_MID_rating", "away_MID_rating", "home_FWD_rating", "away_FWD_rating"]
    

class HeadToHeadWinningRateFeature(Feature):
    def __init__(self, games_df):
        self.games_df = games_df

    def compute(self, row, is_test):
        year = 2015 if is_test else row['year']
        return self.get_head_to_head_winning_rate(row['home_team_id'], row['away_team_id'], year, row.name)

    def get_head_to_head_winning_rate(self, team1_id, team2_id, year, index):
        h2h_games = self.games_df[
            (((self.games_df['home_team_id'] == team1_id) & (self.games_df['away_team_id'] == team2_id)) |
             ((self.games_df['home_team_id'] == team2_id) & (self.games_df['away_team_id'] == team1_id))) &
            (self.games_df['year'] <= year)
        ]
        
        if h2h_games.empty:
            h2h_games = self.games_df[
                ((self.games_df['home_team_id'] == team1_id) & (self.games_df['away_team_id'] == team2_id)) |
                ((self.games_df['home_team_id'] == team2_id) & (self.games_df['away_team_id'] == team1_id))
            ]
        
        h2h_games = h2h_games.drop(index) if index in h2h_games.index else h2h_games
        
        team1_wins = ((h2h_games['home_team_id'] == team1_id) & (h2h_games['is_home_winner'] == True)).sum() + \
                     ((h2h_games['away_team_id'] == team1_id) & (h2h_games['is_home_winner'] == False)).sum()
        
        total_games = len(h2h_games)
        
        return [team1_wins / total_games if total_games > 0 else 0.5]

    def get_feature_names(self):
        return ['home_team_head_to_head_winning_rate']
    
class WorkRateByPositionFeature(Feature):
    def __init__(self, players_df):
        self.players_df = players_df

    def compute(self, row, is_test):
        year = 2015 if is_test else row['year']
        home_attacking, home_defensive = self.get_work_rate_averages(row, 'home', year)
        away_attacking, away_defensive = self.get_work_rate_averages(row, 'away', year)
        return [home_attacking, home_defensive, away_attacking, away_defensive]

    def get_work_rate_averages(self, row, team, year):
        attacking_rates = []
        defensive_rates = []
        for i in range(1, 12):
            player_id = row[f'{team}_player_{i}']
            y_coord = row[f'{team}_player_Y{i}']
            if pd.notna(player_id) and pd.notna(y_coord):
                player_data = self.players_df[(self.players_df['player_id'] == player_id) & 
                                              (self.players_df['year'] <= year)].sort_values('year', ascending=False).iloc[0]
                if 6 <= y_coord <= 10:
                    attacking_rates.append(player_data['attacking_work_rate'])
                else:
                    defensive_rates.append(player_data['defensive_work_rate'])
        
        avg_attacking = np.mean(attacking_rates) if attacking_rates else 1
        avg_defensive = np.mean(defensive_rates) if defensive_rates else 1
        return avg_attacking, avg_defensive

    def get_feature_names(self):
        return ["home_attacking_work_rate", "home_defensive_work_rate", 
                "away_attacking_work_rate", "away_defensive_work_rate"]
        
class TeamStatsFeature(Feature):
    def __init__(self, players_df, player_stats):
        self.players_df = players_df
        self.player_stats = player_stats

    def compute(self, row, is_test):
        year = 2015 if is_test else row['year']
        home_stats = self.get_team_stats(row, 'home', year)
        away_stats = self.get_team_stats(row, 'away', year)
        return home_stats + away_stats

    def get_team_stats(self, row, team, year):
        team_stats = {stat: [] for stat in self.player_stats}
        
        for i in range(1, 12):
            player_id = row[f'{team}_player_{i}']
            if pd.notna(player_id):
                player_data = self.players_df[(self.players_df['player_id'] == player_id) & 
                                              (self.players_df['year'] <= year)].sort_values('year', ascending=False).iloc[0]
                for stat in self.player_stats:
                    team_stats[stat].append(player_data[stat])
        
        stats_summary = []
        for stat in self.player_stats:
            if team_stats[stat]:
                stats_summary.extend([np.mean(team_stats[stat]), np.std(team_stats[stat])])
            else:
                stats_summary.extend([np.nan, np.nan])
        
        return stats_summary

    def get_feature_names(self):
        return [f"{team}_{stat}_{metric}" 
                for team in ['home', 'away'] 
                for stat in self.player_stats 
                for metric in ['mean', 'std']]

class TeamHistoricalWinningRateFeature(Feature):
    def __init__(self, games_df):
        self.games_df = games_df

    def compute(self, row, is_test):
        year = 2015 if is_test else row['year']
        
        home_team_winning_rate = self.get_team_historical_winning_rate(row['home_team_id'], year, row.name)    
        away_team_winning_rate = self.get_team_historical_winning_rate(row['away_team_id'], year, row.name)
        
        return list(home_team_winning_rate) + list(away_team_winning_rate)

    def get_team_historical_winning_rate(self, team_id, year, index):
        # Get the historical games
        home_games = self.games_df[(self.games_df['home_team_id'] == team_id) & (self.games_df['year'] <= year)]
        away_games = self.games_df[(self.games_df['away_team_id'] == team_id) & (self.games_df['year'] <= year)]
        
        home_games = home_games.drop(index) if index in home_games.index else home_games
        away_games = away_games.drop(index) if index in away_games.index else away_games
        
        # Get all games if no history
        if len(home_games) == 0 or len(away_games) == 0:
            home_games = self.games_df[(self.games_df['home_team_id'] == team_id)]
            away_games = self.games_df[(self.games_df['away_team_id'] == team_id)]
            
        # Use away games for home games if no home games and vice versa
        if len(home_games) == 0:
            home_games = self.games_df[(self.games_df['away_team_id'] == team_id)]
            
        if len(away_games) == 0:
            away_games = self.games_df[(self.games_df['home_team_id'] == team_id)]
        
        # Default values if no games
        if len(home_games) == 0 or len(away_games) == 0:
            return np.array([0.5, 0.5, 0.5])    
        
        home_games_winning = home_games[home_games['is_home_winner'] == True]
        away_games_winning = away_games[away_games['is_home_winner'] == False]
            
        home_games_winning_rate = len(home_games_winning) / len(home_games)
        away_games_winning_rate = len(away_games_winning) / len(away_games)
        
        total_winning_rate = (len(home_games_winning) + len(away_games_winning)) / (len(home_games) + len(away_games))
        
        return np.array([home_games_winning_rate, away_games_winning_rate, total_winning_rate])
    
    def get_feature_names(self):
        return ['home_team_home_winning_rate', 'home_team_away_winning_rate', 'home_team_total_winning_rate', 'away_team_home_winning_rate', 'away_team_away_winning_rate', 'away_team_total_winning_rate']
