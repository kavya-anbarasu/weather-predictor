import matplotlib.pyplot as plt
import numpy as np

class Predictor(object):
	def __init__(self):
		'''
		Initialize the values that are necessary for making predictions:
			temp_calendar: annual trend
			slope and intercept: of the three day relationship
		'''
		temps, year, month, day = self.get_data()
		self.temp_calendar = self.build_temp_calendar(temps, year, month, day)
		deseasonalized_temps = np.zeros(temps.size)
		for i, temp in enumerate(temps):
			seasonal_temp = self.find_seasonal_temp(year[i], month[i], day[i])
			deseasonalized_temps[i] = temp - seasonal_temp
		self.slope, self.intercept = self.get_three_day_coefficientes(deseasonalized_temps)
			
	def get_data(self):
		'''
		Parameters
		----------
		None

		Returns
		----------
		temps: array of floats
		year, month, day: array of ints
		'''
		# Read in weather data
		weather_filename = "fortLauderdale.csv"
		weather_file = open(weather_filename)
		weather_data = weather_file.read()
		weather_file.close()

		# Break the weather data into lines
		lines = weather_data.split('\n')
		labels = lines[0]
		values = lines[1:]
		n_values = len(values)

		# Break the list of comma-separated value strings into lists of values.
		year = []
		month = []
		day = []
		max_temp = []
		j_year = 1
		j_month = 2
		j_day = 3
		j_max_temp = 5

		for i_row in range(n_values):
			split_values = values[i_row].split(',')
			if len(split_values) >= j_max_temp:
				year.append(int(split_values[j_year]))
				month.append(int(split_values[j_month]))
				day.append(int(split_values[j_day]))
				max_temp.append(float(split_values[j_max_temp]))

		# Isolate the recent data
		i_mid = len(max_temp) // 2
		temps = np.array(max_temp[i_mid:])
		year = year[i_mid:]
		month = month[i_mid:]
		day = day[i_mid:]
		temps[np.where(temps == -99.9)] = np.nan

		# Remove all the nan's
		# Trim both ends and fill nans in the middle
		# Find the first non-nan
		i_start = np.where(np.logical_not(np.isnan(temps)))[0][0]
		temps = temps[i_start:]
		year = year[i_start:]
		month = month[i_start:]
		day = day[i_start:]
		i_nans = np.where(np.isnan(temps))[0]

		# Replace all nans with the most recent non-nan
		for i in range(temps.size):
			if np.isnan(temps[i]):
				temps[i] = temps[i - 1]
		return (temps, year, month, day)

	# Determine whether the previous day's temperature is related to that of the following day
	# Show the relationship between two variables adding some jitter

	def scatter(self, x, y):
		# Make a scatter plot with jitter
		x_jitter = x + np.random.normal(size = x.size, scale = .5)
		y_jitter = y + np.random.normal(size = y.size, scale = .5)
		plt.plot(x_jitter, y_jitter, color = "black", marker = ".", linestyle = "none", alpha = 0.05)
		plt.show()

	def find_autocorr(self, values, length = 100):
		'''
		Parameters
		----------
		values: arrays of floats
		length: int
			The number of shifts to calculate, the maximum offseat in number of samples
		Returns
		-------
		autocorr: array of floats
		'''
		autocorr = []
		for shift in range(1, length):
			correlation = (np.corrcoef(values[:-shift], values[shift:])[1, 0])
			autocorr.append(correlation)
		return autocorr

		# Add a sine wave apporximation
		d = np.arange(1000)
		fit = .6 * np.cos(2 * np.pi * d / 365)
		#plt.plot(d, fit, color = "green")
		#plt.show()

	def find_day_of_year(self, year, month, day):
		''' 
		Convert year, month, and date to day of the year
		Jan 1 = 0

		Parameters
		----------
		year: int
		month: int
		day: int

		Returns
		-------
		day_of_year: int
		'''
		days_per_month = np.array([
			31, # January
			28, # February
			31, # March
			30, # April
			31, # May
			30, # June
			31, # July
			31, # August
			30, # September
			31, # October
			30, # November
			31, # December
		])
		# For leap years:
		if year % 4 == 0:
			days_per_month[1] += 1

		day_of_year = np.sum(np.array(days_per_month[:month - 1])) + day - 1
		return day_of_year



	def build_temp_calendar(self, temps, year, month, day):
		'''
		Create an array of median temperature by day-of-year
		Day 0 = Jan 1 etc.
		
		Parameters
		----------
		temps: array of floats
		year, month, day: array on ints

		Returns
		-------
		temp_calendar: array of floats of length 366
		'''
		day_of_year = np.zeros(temps.size)
		for i_row in range(temps.size):
			day_of_year[i_row] = self.find_day_of_year(year[i_row], month[i_row], day[i_row])

		# Create 10-day medians for each day of the year
		median_temp_calendar = np.zeros(366)
		for i_day in range (0, 365):
			low_day = i_day - 5
			high_day = i_day + 4
			if low_day < 0:
				low_day += 365
			if high_day > 365:
				high_day -365
			if low_day < high_day:
				i_windows_days = np.where(
					np.logical_and(day_of_year >= low_day, day_of_year <= high_day))
			else:
				i_windows_days = np.where(
					np.logical_or(day_of_year >= low_day, day_of_year <= high_day))

			ten_day_median = np.median(temps[i_windows_days])
			median_temp_calendar[i_day] = ten_day_median

			if i_day == 364:
				median_temp_calendar[365] = ten_day_median
		return median_temp_calendar

	def get_three_day_coefficientes(self, deseasonalized_temps):
		'''
		Parameters
		----------
		deseasonalized_temps: array of floats
		
		Returns
		-------
		slope, intercept: floats
			coefficients of the line showing the relationship
			between deseasonalized temperatures and those
			three days into the future
		'''
		slope, intercept = np.polyfit(deseasonalized_temps[:-3], deseasonalized_temps[3:], 1)
		return (slope, intercept)

	def find_seasonal_temp(self, year, month, day):
		'''
		For a given day, month, and year, find the seasonal high temperature
		for Fort Lauderdale Beach.

		Parameters
		----------
		year, month, day: int
			The day of interest
		
		Returns
		-------
		seasonal_temp: float
		'''
		doy = self.find_day_of_year(year, month, day)
		seasonal_temp = self.temp_calendar[doy]
		return seasonal_temp

	def predict_deseasonalized(self, three_day_temp):
		'''
		Based on a deseasonalized temperature, predict_deseasonalized what the 
		deseasonalized temperature will be three day in the future.

		Parameters
		----------
		three_day_temp: float
			The measured temperature three days before the day of interest. 
		Results
		-------
		predicted_temp: float
		'''
		predicted_temp = self.intercept + self.slope * three_day_temp
		return predicted_temp
	
	def deseasonalize(self, temp, doy):
		'''
		Deseasonalize a temperature by subtracting out the anual trend
		Parameters
		----------
		temp: float
		doy: int

		Return
		-------
		deseasonalized_temp: float
		'''
		deseasonalized_temp = temp - self.temp_calendar[doy]
		return deseasonalized_temp

	def reseasonalize(self, deseasonalized_temp, doy):
		'''
		Reconstitute the deseasonalized temperature by adding back in the anual trend
		Parameters
		----------
		deseasonalized_temp: float
		doy: int

		Return
		-------
		reseasonalized_temp: float
		'''
		reseasonalized_temp = deseasonalized_temp + self.temp_calendar[doy]
		return reseasonalized_temp

	def predict(self, year, month, day, past_temp):
		'''
		Parameters
		----------
		year, month, day: ints
		past_temp: float
			The temperature from 3 days before the date of interest
		'''
		
		# Make a prediction for a single day.
		doy = self.find_day_of_year(year, month, day)
		doy_past = doy - 3
		if doy_past < 0:
			doy_past += 365

		deseasonalized_temp = self.deseasonalize(past_temp, doy_past)
		# Make predictions for three days into the future
		deseasonalized_prediction = self.predict_deseasonalized(deseasonalized_temp)
		prediction = self.reseasonalize(deseasonalized_prediction, doy)

		return prediction

def test():
	'''
	Run through the data history, calculating the prediction
	error for each day.
	'''
	predictor = Predictor()

	temps, year, month, day = predictor.get_data()
	deseasonalized_temps = np.zeros(temps.size)
	doy = np.zeros(temps.size, dtype = np.int)
	for i, temp in enumerate(temps):
		seasonal_temp = predictor.find_seasonal_temp(year[i], month[i], day[i])
		deseasonalized_temps[i] = temp - seasonal_temp
		doy[i] = predictor.find_day_of_year(year[i], month[i], day[i])
	# Make predictions for three days into the future
	deseasonalized_predictions = np.zeros(temps.size)
	for i, temp in enumerate(deseasonalized_temps):
		deseasonalized_predictions[i] = predictor.predict_deseasonalized(temp)
	
	predictions = np.zeros(temps.size - 3)
	for i, temp in enumerate(deseasonalized_predictions[:-3]):
		predictions[i] = predictor.reseasonalize(temp, doy[i + 3])

	residuals = temps[3:] - predictions
	print("MEA:", np.mean(np.abs(residuals)))

	actuals = temps[3:]

	# Compare predictions three days away vs. actuals for above/below 85
	sensitivity = []
	targets = np.arange(84, 90)
	for target in targets:
		i_warm = np.where(actuals > 85)[0]
		i_warm_predictions = np.where(predictions > target)[0]
		n_true_positives = np.intersect1d(i_warm, i_warm_predictions).size
		n_false_negatives = np.setdiff1d(i_warm, i_warm_predictions).size
		n_false_positives = np.setdiff1d(i_warm_predictions, i_warm).size
		n_true_negatives = (actuals.size - n_true_positives - n_false_positives - n_false_negatives)

		#print("Accurately predicted warm", n_true_positives, "times")
		#print("Predicted cold when it was warm", n_false_negatives, "times")
		#print("Predicted warm when it was cold", n_false_positives, "times")
		#print("Accurately predicted cold", n_true_negatives, "times")

		sensitivity.append(n_true_positives / (n_true_positives + n_false_positives))
		#print("Fraction of warm trips", sensitivity[-1])

def test_single():
	'''
	Make a single prediction
	'''
	# The day for which to predict_deseasonalized
	test_year = 2014
	test_month = 7 
	test_day = 17
	test_temp = 86

	prediction = predictor.predict(test_year, test_month, test_day, test_temp)

	print("For year = ", test_year, ", month = ", test_month, ", day = ", test_day, ", temp = ", test_temp)
	print("predicted temperature is ", prediction)


if __name__ == '__main__':
	predictor = Predictor()
	test()
	test_single()
