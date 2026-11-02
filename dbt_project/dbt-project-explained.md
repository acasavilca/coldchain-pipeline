## analyses

- A place for SQL files that you don't want exposed
- Can be used for quality reports
- Or not used at all

## dbt_project.yml

- Most important file in dbt
- Tells dbt some defaults
- Need it for running dbt commands
- For dbt core, profile must match the one in `.dbt/profiles.yml`

## macros

- They behave like Python functions (reusable logic)
- They help encapsulate logic in one place

## README.md

- Documentation
- Installation/setup guides
- Contact information
- Overall, information on how to run a project

## seeds

- Need to ingest data into database, but aren't ready to create a table or have the right permissions
- A space to upload csv and flat files to add them to dbt later
- Lookup tables, little tests (quick and dirty)

## snapshots

- Takes a picture of table at a certain moment in time
- Useful to track history of column that overwrites itself

## tests

- A place to put assertions in SQL format
- Singular tests
- If SQL command returns more than 0 rows, the dbt build fails

## models

### staging
- Sources (raw table from database)
- Staging files are 1 to 1 copy of your data with minimal cleaning steps:
    - Data types
    - Renaming columns

### intermediate
- Anything not raw but not to be exposed either
- No guidelines, just nice for heavy duty cleaning or complex logic

### marts
- If it's in marts, it's ready for consumption
- Tables ready for dashboards
- Properly modeled, clean tables