import numpy as np
import pandas as pd
import operator as op
import math
import csv


class Emblem:
	def __init__(self, filename, sep=',', debug=False):
		self.debug = debug
		# READ INPUT FILENAME ---------------------------------------------------------------------
		self.file = []
		with open(filename, 'rt') as csvfile:
			spamreader = csv.reader(csvfile, quotechar='"', delimiter=sep)
			for row in spamreader:
				self.file.append(np.array(row).astype(str))
		self.file_len = len(self.file)

		# Coefficients ----------------------------------------------------------------------------
		self.base_weight = 0
		self.univariate_beta = {}
		self.single_univariate_beta = {}
		self.bivariate_beta = {}
		self.bivariate_beta_flat = {}
		self.end_bivariates_reached = False

		# Rows ------------------------------------------------------------------------------------
		self.start_univariate_row = 0
		self.last_univariate_row = 0

		# Variables ------------------------------------------------------------------------------------
		self.univariate_variables = []
		self.bivariate_variables = []
		self.read_base_level()
		self.read_univariates()
		self.read_bivariates()
		self.flatten_bivariates()

	# BASE LEVEL ----------------------------------------------------------------------------------
	def read_base_level(self):

		self.base_weight = float(self.file[0][2])
		self.start_univariate_row = 5

	# UNIVARIATES ----------------------------------------------------------------------------------
	def read_univariates(self):
		univariate_levels_cols = np.array(np.where(self.file[self.start_univariate_row] != '')[0])
		self.univariate_beta = {}
		self.last_univariate_row = 0

		for col in univariate_levels_cols:
			self.single_univariate_beta = {}
			row = self.start_univariate_row + 1
			single_univariate_end_reached = False
			univariate_name = self.file[row - 1][col]
			if self.debug:
				print('{:=<100}'.format('{} '.format(univariate_name)))

			while not single_univariate_end_reached:

				if self.file_len == row + 1:
					single_univariate_end_reached = True
					self.end_bivariates_reached = True
				elif self.file[row + 1][col + 1] == '':
					single_univariate_end_reached = True
				try:
					level = float(self.file[row][col])
				except ValueError:
					level = self.file[row][col]

				if self.debug:
					print('\trow: {:<6}| col: {:<6}| level: {:<40}| value: {:<8}'.format(row, col, level, self.file[row][col + 1]))

				self.single_univariate_beta.update({level: float(self.file[row][col + 1])})
				row += 1

			self.univariate_beta.update({univariate_name: self.single_univariate_beta})
			self.last_univariate_row = max(self.last_univariate_row, row)

			self.univariate_variables = list(self.univariate_beta.keys())

	def read_bivariates(self):
		row = self.last_univariate_row + 2
		self.bivariate_beta = {}

		if not self.end_bivariates_reached:
			if not (self.file[row - 2][0] == 'Orthogonal Polynomial Equations' or self.file[row - 1][0] == 'Orthogonal Polynomial Equations' or self.file[row][0] == 'Orthogonal Polynomial Equations'):
				# cycle over bivariate matrices
				while not self.end_bivariates_reached:
					first_bivariate_name = self.file[row + 2][0]
					second_bivariate_name = self.file[row][2]
					bivariate_interaction_name = first_bivariate_name + '|' + second_bivariate_name

					try:
						second_bivariate_levels = list(map(float, self.file[row + 1][np.where(self.file[row + 1] != '')]))
					except ValueError:
						second_bivariate_levels = self.file[row + 1][np.where(self.file[row + 1] != '')]

					row += 2

					# cycle over bivariate matrix's rows
					single_bivariate_end_reached = False
					single_bivariate_beta = {}
					while not single_bivariate_end_reached:
						try:
							first_bivariate_level = float(self.file[row][1])
						except ValueError:
							first_bivariate_level = self.file[row][1]
						first_bivariate_level = first_bivariate_level if str(first_bivariate_level) != '' else 'nan'

						single_bivariate_beta.update({first_bivariate_level: dict(zip(second_bivariate_levels, list(map(float, self.file[row][2:2 + len(second_bivariate_levels)]))))})

						if row >= self.file_len - 1:
							single_bivariate_end_reached = True
							self.end_bivariates_reached = True
						elif self.file[row + 1][2] == '':
							single_bivariate_end_reached = True

							if row < self.file_len - 1:
								if self.file[row + 2][0] == 'Orthogonal Polynomial Equations':
									self.end_bivariates_reached = True

						row += 1

					row += 3

					self.bivariate_beta.update({bivariate_interaction_name: single_bivariate_beta})

				self.bivariate_variables = list(set(np.array(list(map(lambda x: x.split('|'), self.bivariate_beta.keys()))).flatten()))

	def flatten_bivariates(self):
		self.bivariate_beta_flat = {}
		for interaction_name, bivariate_dict in self.bivariate_beta.items():
			flat_dict = pd.DataFrame(bivariate_dict).unstack()
			self.bivariate_beta_flat.update({interaction_name: {'{}|{}'.format(k[0], k[1]): v for k, v in flat_dict.items()}})

	def linear_predictor(self, df: pd.DataFrame, excluded_vars=None, debug=False) -> np.array:
		return self.evaluate(df=df, operator=op.add, excluded_vars=excluded_vars, debug=debug)

	def evaluate(self, df: pd.DataFrame, operator, excluded_vars: list=None, debug=False) -> np.array:
		if excluded_vars is None:
			excluded_vars = []

		df = df.reset_index(drop=True)
		result = np.array([self.base_weight for _ in range(len(df))])

		for variable, beta_dict in self.univariate_beta.items():
			if variable not in excluded_vars:
				if debug:
					try:
						result = operator(result, df[variable].apply(lambda x: beta_dict[x]))
					except KeyError as ke:
						print(variable, ke)
						raise KeyError
				else:
					result = operator(result, df[variable].map(beta_dict))

		for interaction_name, interaction_dict in self.bivariate_beta_flat.items():
			first_bivariate = interaction_name.split('|')[0]
			second_bivariate = interaction_name.split('|')[1]

			first_bivariate_values = df[first_bivariate]
			second_bivariate_values = df[second_bivariate]

			if not ((first_bivariate in excluded_vars) or (second_bivariate in excluded_vars)):
				try:
					first_bivariate_values = np.round(first_bivariate_values.astype(float), 3)
				except ValueError:
					pass

				try:
					second_bivariate_values = np.round(second_bivariate_values.astype(float), 3)
				except ValueError:
					pass

				compact_bivariate = first_bivariate_values.astype(str) + pd.Series(['|' for _ in range(len(df))]) + second_bivariate_values.astype(str)

				if debug:
					try:
						result = operator(result, compact_bivariate.apply(lambda x: interaction_dict[x]))
					except KeyError as ke:
						try:
							ke = float(str(ke))
						except TypeError:
							pass

						if math.isnan(ke):
							error_condition = compact_bivariate.apply(lambda x: math.isnan(x) if isinstance(x, float) else False)
						else:
							error_condition = compact_bivariate.isin([ke])

						print(interaction_name, ke, '\n')
						print(pd.concat([first_bivariate_values, second_bivariate_values, compact_bivariate], axis=1)[error_condition].iloc[0:10, :])
						raise KeyError
				else:
					result = operator(result, compact_bivariate.map(interaction_dict))

		null_values_index = result.isna()
		null_values = null_values_index.sum()

		if result.isna().sum() > 0:
			print('Warning! {} null values found in result'.format(null_values))

		return np.array(result.reset_index(drop=True))

	def predict(self, df: pd.DataFrame, link_function, excluded_vars=None, debug=False) -> pd.Series:
		link_function = np.vectorize(link_function)
		return pd.Series(data=link_function(self.linear_predictor(df, excluded_vars, debug)), index=df.index)

	def get_beta(self, df: pd.Series, index_variable=None):
		beta_values = []
		for idx, row in df.iterrows():
			beta_values_tmp = {
				'index': idx if index_variable is None else row[index_variable],
				'base_weight': self.base_weight,
				'weights': {}
			}
			for variable, beta_dict in self.univariate_beta.items():
				beta_values_tmp['weights'].update({variable: {row[variable]: beta_dict[row[variable]]}})

			for interaction_name, interaction_dict in self.bivariate_beta.items():
				first_bivariate = interaction_name.split('|')[0]
				second_bivariate = interaction_name.split('|')[1]
				beta_values_tmp['weights'].update({
					'{}|{}'.format(first_bivariate, second_bivariate): {'{}|{}'.format(row[first_bivariate], row[second_bivariate]): interaction_dict[row[first_bivariate]][row[second_bivariate]]}
				})

			beta_values.append([beta_values_tmp])

		return pd.DataFrame(data=beta_values, index=df.index, columns=['beta_values'])
