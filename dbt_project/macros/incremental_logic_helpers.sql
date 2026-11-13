{%- macro ensure_checkpoint_table(model_relation, prefix='ck_') %}
    {% if execute %}
        {% set identifier = prefix ~ model_relation.identifier %}
        {% set fix = "run: dbt run --full-refresh -s " ~ model_relation.identifier ~ "+" %}
        {% set cp_relation = adapter.get_relation(
            database=model_relation.database,
            schema=model_relation.schema,
            identifier=identifier
        ) %}

        {% if cp_relation is none %}
            {{ exceptions.raise_compiler_error("Checkpoint table '" ~ identifier ~ "' is missing; " ~ fix) }}
        {% endif %}

        {% set row_count = run_query("select count(*) from " ~ cp_relation).columns[0].values()[0] | int %}
        {% if row_count == 0 %}
            {{ exceptions.raise_compiler_error("Checkpoint table '" ~ identifier ~ "' is empty; " ~ fix) }}
        {% endif %}

        {{ return(cp_relation) }}
    {% endif %}
{%- endmacro %}

{#-
  ingest_watermark(relation, ts_col, hours=6, ts_type='timestamp')
  Returns a timestamp string = max(ingested_at) of `relation` minus `hours`, read only from the
  last two partition days (ts_col must be the table's partition column; ts_type='date' for DATE partitions).
  Returns var('cutoff_override') when set, and the epoch when the table has no data (= full rebuild).
-#}
{%- macro ingest_watermark(relation, ts_col, hours=6, ts_type='timestamp') %}
    {% if var('cutoff_override', none) %}
        {{ return(var('cutoff_override')) }}
    {% endif %}
    {% if execute %}
        {# 1. Find the latest partition with rows #}
        {% set parts_sql %}
            select cast(parse_date('%Y%m%d', max(partition_id)) as string)
            from `{{ relation.database }}`.`{{ relation.schema }}`.INFORMATION_SCHEMA.PARTITIONS
            where table_name = '{{ relation.identifier }}'
              and partition_id not in ('__NULL__', '__UNPARTITIONED__')
              and total_rows > 0
        {% endset %}
        {# 2. Convert it to a day #}
        {% set latest_day = run_query(parts_sql).columns[0].values()[0] %}
        {% if latest_day is none %}
            {{ return('1970-01-01 00:00:00+00:00') }}
        {% endif %}
        {# 3. Find the anchor: max(ingested_at) over the last two partition days #}
        {% if ts_type == 'date' %}
            {% set floor_expr = "date_sub(date('" ~ latest_day ~ "'), interval 1 day)" %}
        {% else %}
            {% set floor_expr = "timestamp_sub(timestamp('" ~ latest_day ~ "'), interval 1 day)" %}
        {% endif %}
        {% set wm_sql %}
            select cast(timestamp_sub(coalesce(max(ingested_at), timestamp('1970-01-01')), interval {{ hours }} hour) as string)
            from {{ relation }}
            where {{ ts_col }} >= {{ floor_expr }}
        {% endset %}
        {{ return(run_query(wm_sql).columns[0].values()[0]) }}
    {% endif %}
    {{ return('1970-01-01 00:00:00+00:00') }}
{%- endmacro %}
