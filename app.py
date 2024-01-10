import sqlvalidator
from valentine.algorithms import cupid
from model.language import siamese_model, compare_queries
from model.compare import compare

questions = {
  "Write a query that returns employee_Id, name, salary of all employees, and additionally a new_salary. The intermediate column new_salary shows the current salary increased by 15%. The new salary must be returned as integer value.": "Select ID, Name, salary, ROUND(salary*1.15,0) AS New_Salary From employee;",
  "Create a query that returns following string for every employee: < name > earns < salary > per month, but desires to earn < 3*salary >. Replace all placeholders with the respective data using SQL. The new column is called desired salary.": "select concat(name, ' earns ' ,salary , ' per month, but wants to have ' , salary * 3) AS dream_salary from employee;",
  "List all plane types. Thereby, all first letters must be capitalized, the rest must be uncapitalized. Return the length of the type name in a second column. The columns are called name und length.": "select count(*) as Amount from (select distinct serial_number from departure) as a"
}

def is_sql_valid(query):
  sql_query = sqlvalidator.parse(query)

  if not sql_query.is_valid():
    return False, sql_query.errors
  
  return True, []


def main():
  # Example usage
  question = "What is the salary information of employees?"
  query1 = "Select ID, Name, salary, ROUND(salary*1.15,0) AS New_Salary From employee;"
  query2 = "Select ID, Name, salary, ROUND(salary*1.15,0) AS New_Salary From employee;"

  is_valid, errors = is_sql_valid(query2)

  if not is_valid:
    print("Syntax Error")
    return ""
  
  is_syntax_equivalent, errors = compare(query1, query2)

  print(is_syntax_equivalent)

  if not is_syntax_equivalent:
     print(errors)
     return ""

  similarity_score = compare_queries(question, query1, query2, siamese_model)
  print(f"Similarity Score: {similarity_score}")

  threshold = 0.5  # Adjust the threshold based on your requirements
  if similarity_score >= threshold:
      print("Queries are likely functionally equivalent.")
  else:
      print("Queries are likely not functionally equivalent.")


if __name__ == '__main__':
  main()