import numpy
import csv


class Emblem:
	def __init__(self, filename):
		# READ INPUT FILENAME ---------------------------------------------------------------------
		self.file = []
		with open(filename, 'rt') as csvfile:
			spamreader = csv.reader(csvfile, quotechar='"', delimiter=',')
			for row in spamreader:
				self.file.append(numpy.array(row).astype(str))
		self.file_len = len(self.file)

		# Coefficients ----------------------------------------------------------------------------
		self.base_weight = 0
		self.univariate_beta = {}
		self.single_univariate_beta = {}
		self.bivariate_beta = {}

		# Rows ------------------------------------------------------------------------------------
		self.start_univariate_row = 0
		self.last_univariate_row = 0

		self.read_base_level()
		self.read_univariates()
		self.read_bivariates()

	# BASE LEVEL ----------------------------------------------------------------------------------
	def read_base_level(self):
		self.base_weight = float(self.file[0][2])
		self.start_univariate_row = 5

	# UNIVARIATES ----------------------------------------------------------------------------------
	def read_univariates(self):
		univariate_levels_cols = numpy.array(numpy.where(self.file[self.start_univariate_row] != '')[0])
		self.univariate_beta = {}
		self.last_univariate_row = 0

		for col in univariate_levels_cols:
			self.single_univariate_beta = {}
			row = self.start_univariate_row + 1
			single_univariate_end_reached = False
			univariate_name = self.file[row - 1][col]

			while not single_univariate_end_reached:
				if self.file[row][col] == '':
					if len(self.single_univariate_beta) > 0:
						single_univariate_end_reached = True
						row += 1
					else:
						self.single_univariate_beta.update({'nan': float(self.file[row][col + 1])})
						row += 1
				elif self.file[row + 1][col] == '' and self.file[row][col + 1] == '':
					single_univariate_end_reached = True
					row += 1
				else:
					try:
						level = float(self.file[row][col])
					except ValueError:
						level = self.file[row][col]

					self.single_univariate_beta.update({level: float(self.file[row][col + 1])})
					row += 1

			self.univariate_beta.update({univariate_name: self.single_univariate_beta})
			self.last_univariate_row = max(self.last_univariate_row, row)

	def read_bivariates(self):
		row = self.last_univariate_row + 1
		self.bivariate_beta = {}

		# cycle over bivariate matrices
		end_bivariates_reached = False
		while not end_bivariates_reached:
			first_bivariate_name = self.file[row + 2][0]
			second_bivariate_name = self.file[row][2]
			bivariate_interaction_name = first_bivariate_name + '|' + second_bivariate_name

			try:
				second_bivariate_levels = list(map(float, self.file[row + 1][numpy.where(self.file[row + 1] != '')]))
			except ValueError:
				second_bivariate_levels = self.file[row + 1][numpy.where(self.file[row + 1] != '')]

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
					end_bivariates_reached = True
				elif self.file[row + 1][2] == '':
					single_bivariate_end_reached = True

					if row < self.file_len - 1:
						if self.file[row + 2][0] == 'Orthogonal Polynomial Equations':
							end_bivariates_reached = True

				row += 1

			row += 3

			self.bivariate_beta.update({bivariate_interaction_name: single_bivariate_beta})
