import sqlparse
import pandas as pd
from valentine.algorithms import Coma, JaccardDistanceMatcher, DistributionBased, SimilarityFlooding, Cupid
from valentine.data_sources import DataframeTable
from valentine.algorithms.jaccard_distance import StringDistanceFunction

def extract_column_or_expression(token):
    """
    Extracts a column, SQL function, or arithmetic expression from the token.
    """
    if any(op in token.value for op in ["(", "+", "-", "*", "/"]):
        # Treat as a function or arithmetic expression
        return token.value
    else:
        # Assume it's a column name
        return token.get_real_name()

def parse_sql_query(query):
    parsed = sqlparse.parse(query)[0]
    table_name = None
    columns = []
    is_selecting_columns = False
    from_seen = False
    statement_type = None

    for token in parsed.tokens:
        if token.ttype is sqlparse.tokens.DML:
            statement_type = token.value.upper()  # Capture the SQL statement type
            if statement_type == 'SELECT':
                is_selecting_columns = True
        elif token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
            from_seen = True
        elif from_seen and isinstance(token, sqlparse.sql.Identifier):
            table_name = token.get_real_name()
            break
        elif is_selecting_columns:
            if isinstance(token, sqlparse.sql.IdentifierList):
                for identifier in token.get_identifiers():
                    columns.append(extract_column_or_expression(identifier))
            elif isinstance(token, (sqlparse.sql.Identifier, sqlparse.sql.Function)):
                columns.append(extract_column_or_expression(token))

    return statement_type, table_name, columns

def generate_dataframes(query1, query2):
    df1 = pd.DataFrame([parse_sql_query(query1)], columns=['statement_type', 'table_name', 'columns'])
    df2 = pd.DataFrame([parse_sql_query(query2)], columns=['statement_type', 'table_name', 'columns'])

    return df1, df2

def compare_column_values(values1, values2):
    """
    Compare the values in two lists and return a similarity score.
    """
    common_values = set(values1) & set(values2)
    total_values = set(values1) | set(values2)
    return len(common_values) / len(total_values) * 100 if total_values else 0

def compare_all_columns(df1, df2):
    """
    Compare all corresponding columns in two dataframes and print similarity scores.
    """
    scores = {}
    for column in df1.columns:
        if column in df2.columns:
            score = compare_column_values(df1[column][0], df2[column][0])
            scores[column] = score
            # print(f"Column '{column}' match score: {score:.2f}%")
        else:
            # print(f"Column '{column}' does not exist in both dataframes.")
            pass
    return scores

def compare(query1, query2):
    df1, df2 = generate_dataframes(query1, query2)
    scores = compare_all_columns(df1, df2)

    # Errors
    errors = []

    for column in scores:
        if scores[column] < 70:
            column_name = column.replace("_", " ", 2).capitalize()
            errors.append(f'Check {column_name}')

    return len(errors) == 0, errors
