{% macro get_sensor_fill_columns(tags) %}
    {%- for tag in tags -%}
        any_value(case when tag_name = '{{ tag }}' then value_filled end) as {{ tag }},
        any_value(case when tag_name = '{{ tag }}' then fill_status end) as {{ tag }}_fill_status{%- if not loop.last -%},{%- endif %}
    {% endfor -%}
{%- endmacro %}

{%- macro fill_sensor_cadence_gaps(tags, partition_key, timestamp_col, unknown_label, table_alias=none) %}
    {%- set prefix = table_alias ~ '.' if table_alias else '' -%}
    {%- for tag in tags -%}
        case
            when last_value({{ prefix }}{{ tag }}_fill_status ignore nulls) over (
                partition by {{ partition_key }} order by {{ timestamp_col }}
                rows between unbounded preceding and current row
            ) = '{{ unknown_label }}' then null
            else last_value({{ prefix }}{{ tag }} ignore nulls) over (
                partition by {{ partition_key }} order by {{ timestamp_col }}
                rows between unbounded preceding and current row
            )
        end as {{ tag }},
        last_value({{ prefix }}{{ tag }}_fill_status ignore nulls) over (
            partition by {{ partition_key }} order by {{ timestamp_col }}
            rows between unbounded preceding and current row
        ) as {{ tag }}_fill_status,
        {{ prefix }}{{ tag }} is null as is_{{ tag }}_cadence_filled{%- if not loop.last -%},{%- endif %}
    {% endfor -%}
{%- endmacro %}

{%- macro fill_state_columns(columns, partition_key, timestamp_col, table_alias=none) %}
    {%- set prefix = table_alias ~ '.' if table_alias else '' -%}
    {%- for col in columns %}
        last_value({{ prefix }}{{ col }} ignore nulls) over (
            partition by {{ partition_key }} order by {{ timestamp_col }}
            rows between unbounded preceding and current row
        ) as {{ col }}{%- if not loop.last -%},{%- endif %}
    {% endfor -%}
{%- endmacro %}

{%- macro fill_sensor_cadence_gaps_window(tags, unknown_label, table_alias=none, window_name='w') %}
    {%- set prefix = table_alias ~ '.' if table_alias else '' -%}
    {%- for tag in tags -%}
        case
            when last_value({{ prefix }}{{ tag }}_fill_status ignore nulls) over {{ window_name }} = '{{ unknown_label }}' then null
            else last_value({{ prefix }}{{ tag }} ignore nulls) over {{ window_name }}
        end as {{ tag }},
        last_value({{ prefix }}{{ tag }}_fill_status ignore nulls) over {{ window_name }} as {{ tag }}_fill_status,
        {{ prefix }}{{ tag }} is null as is_{{ tag }}_cadence_filled{%- if not loop.last -%},{%- endif %}
    {% endfor -%}
{%- endmacro %}

{%- macro fill_state_columns_window(columns, table_alias=none, window_name='w') %}
    {%- set prefix = table_alias ~ '.' if table_alias else '' -%}
    {%- for col in columns %}
        last_value({{ prefix }}{{ col }} ignore nulls) over {{ window_name }} as {{ col }}{%- if not loop.last -%},{%- endif %}
    {% endfor -%}
{%- endmacro %}

{%- macro get_min_polling_interval_ms(tags_list=none) %}
    {% set query -%}
        select min(expected_polling_interval_sec) * 1000 as min_interval_ms
        from {{ ref('seed_sensor_polling_config') }}
        {%- if tags_list %}
        where tag_name in ('{{ tags_list | join("', '") }}')
        {%- endif %}
    {%- endset %}

    {% if execute %}
        {% set result = run_query(query) %}
        {{ return(result.columns[0].values()[0] | int) }}
    {% else %}
        {{ return(100) }}
    {% endif %}
{%- endmacro %}
