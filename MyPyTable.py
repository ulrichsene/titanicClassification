import copy
import csv
from tabulate import tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_instances(self):
        """Computes the dimension of the table (N).

        Returns:
            int: number of rows in the table (N)
        """
        return len(self.data)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = self.column_names.index(col_identifier)
        column = []
        for row in self.data:
            column.append(row[col_index])

        return column

    def fancy_get_column(self, col_identifier):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = self.column_names.index(col_identifier)
        column = []
        for row in self.data:
            column.append([row[col_index]])

        return column

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for i in range(len(row)):
                try:
                    numeric_val = float(row[i])
                    row[i] = numeric_val
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop.sort(reverse=True)
        for row_index in row_indexes_to_drop: #for each loop
            self.data.pop(row_index)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        data_table = []

        infile = open(filename, "r")
        reader = csv.reader(infile)
        for row in reader:
            data_table.append(row)
        self.column_names = data_table.pop(0)
        infile.close()
        self.data = data_table
        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        # table = [self.column_names + self.data]
        # print(table)
        with open(filename, "w", newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)
            outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        unique_rows = []
        duplicates = []
        self.convert_to_numeric()
        for i, row in enumerate(self.data):
            key = []
            for key_column in key_column_names:
                key.append(row[self.column_names.index(key_column)])
            if key in unique_rows:
                duplicates.append(i)
            else:
                unique_rows.append(key)
        return duplicates

    def show_duplicates(self, key_column_names):
        """Returns a list of duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of rows: list of rows of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        unique_rows = []
        duplicates = []
        self.convert_to_numeric()
        for i, row in enumerate(self.data):
            key = []
            for key_column in key_column_names:
                key.append(row[self.column_names.index(key_column)])
            if key in unique_rows:
                duplicates.append(key)
            else:
                unique_rows.append(key)
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        new_data = []
        for row in self.data:
            contains_missing_value = False
            for value in row:
                if value == 'NA':
                    contains_missing_value = True
                if value is None:
                    contains_missing_value = True
            if contains_missing_value is False:
                new_data.append(row)
        self.data = new_data

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)

        valid_values = [float(row[col_index]) for row in self.data if row[col_index] != "NA"]

        average_column_value = sum(valid_values) / len(valid_values)

        for row in self.data:
            if row[col_index] == 'NA':
                row[col_index] = average_column_value

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        table = []
        for col_name in col_names:
            col_index = self.column_names.index(col_name)
            valid_values = [float(row[col_index]) for row in self.data if row[col_index] != "NA"]
            if not valid_values:
                pass
            else:
                col_min = min(valid_values)
                col_max = max(valid_values)
                col_mid = (col_min + col_max) / 2
                col_avg = sum(valid_values) / len(valid_values)

                if len(valid_values) % 2 == 0:
                    median_a = sorted(valid_values)[len(valid_values) // 2]
                    median_b = sorted(valid_values)[len(valid_values) // 2 - 1]
                    col_median = (median_a + median_b) / 2
                else:
                    col_median = sorted(valid_values)[len(valid_values) // 2]

                headers = ["attribute", "min", "max", "mid", "avg", "median"]
                temp_table = [col_name, col_min, col_max, col_mid, col_avg, col_median]
                table.append(temp_table)

        return MyPyTable(headers, table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        self_key_index = [self.column_names.index(col_name) for col_name in key_column_names]
        other_key_index = [other_table.column_names.index(col_name) for col_name in key_column_names]

        joined_data = []
        joined_column_names = self.column_names + [column for column in other_table.column_names if column not in key_column_names]

        for row1 in self.data:
            self_key_values = [row1[i] for i in self_key_index]

            for row2 in other_table.data:
                other_key_values = [row2[i] for i in other_key_index]

                if self_key_values == other_key_values:
                    joined_row = row1 + [row2[i] for i in range(len(row2)) if i not in other_key_index]
                    joined_data.append(joined_row)

        return MyPyTable(column_names=joined_column_names, data=joined_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        self_key_index = [self.column_names.index(col_name) for col_name in key_column_names]
        other_key_index = [other_table.column_names.index(col_name) for col_name in key_column_names]

        joined_column_names = self.column_names + [col for col in other_table.column_names if col not in key_column_names]

        joined_data = []

        for row1 in self.data:
            self_key_values = [row1[i] for i in self_key_index]
            match_value = False

            for row2 in other_table.data:
                other_key_values = [row2[i] for i in other_key_index]

                if self_key_values == other_key_values:
                    joined_row_dict = {col: "NA" for col in joined_column_names}

                    for i, col_name in enumerate(self.column_names):
                        joined_row_dict[col_name] = row1[i]

                    for i, col_name in enumerate(other_table.column_names):
                        if col_name not in key_column_names:
                            joined_row_dict[col_name] = row2[i]

                    joined_row = [joined_row_dict[col] for col in joined_column_names]
                    joined_data.append(joined_row)
                    match_value = True

            if not match_value:
                joined_row_dict = {col: "NA" for col in joined_column_names}

                for i, col_name in enumerate(self.column_names):
                    joined_row_dict[col_name] = row1[i]

                joined_row = [joined_row_dict[col] for col in joined_column_names]
                joined_data.append(joined_row)

        for row2 in other_table.data:
            other_key_values = [row2[i] for i in other_key_index]

            match_value = False

            for row3 in joined_data:
                join_key_values = [row3[joined_column_names.index(col)] for col in key_column_names]

                if other_key_values == join_key_values:
                    match_value = True

            if not match_value:
                joined_row_dict = {col: "NA" for col in joined_column_names}

                for i, col_name in enumerate(other_table.column_names):
                    joined_row_dict[col_name] = row2[i]

                joined_row = [joined_row_dict[col] for col in joined_column_names]
                joined_data.append(joined_row)

        return MyPyTable(joined_column_names, joined_data)
