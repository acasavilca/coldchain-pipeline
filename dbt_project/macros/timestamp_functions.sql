{%- macro quantize_timestamp_ms(ts, quantum_ms) %}
    timestamp_millis(div(unix_millis({{ ts }}), {{ quantum_ms }}) * {{ quantum_ms }})
{%- endmacro %}
