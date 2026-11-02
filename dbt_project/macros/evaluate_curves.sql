{%- macro clip_values(col_name, min_val, max_val) %}
    least(greatest({{ col_name }}, {{ min_val }}), {{ max_val }})
{%- endmacro %}

{%- macro norm_col(col_name, min_val, max_val) %}
    (2 * ({{ col_name }} - {{ min_val }}) / ({{ max_val }} - {{ min_val }}) - 1)
{%- endmacro %}

{%- macro evaluate_poly1d_cubic(x1_col, coeffs_col, bounds_col, clip=true) %}
    {%- if clip -%}
        {%- set x1_val = clip_values(x1_col, bounds_col ~ '[offset(0)]', bounds_col ~ '[offset(1)]') -%}
    {%- else -%}
        {%- set x1_val = x1_col -%}
    {%- endif -%}

    {%- set x1_norm = norm_col(x1_val, bounds_col ~ '[offset(0)]', bounds_col ~ '[offset(1)]') -%}
    (
        {{ coeffs_col }}[offset(0)] +
        {{ coeffs_col }}[offset(1)] * {{ x1_norm }} +
        {{ coeffs_col }}[offset(2)] * ({{ x1_norm }} * {{ x1_norm }}) +
        {{ coeffs_col }}[offset(3)] * ({{ x1_norm }} * {{ x1_norm }} * {{ x1_norm }})
    )
{%- endmacro %}

{%- macro evaluate_poly1d_quadratic(x1_col, coeffs_col, bounds_col, clip=true) %}
    {%- if clip -%}
        {%- set x1_val = clip_values(x1_col, bounds_col ~ '[offset(0)]', bounds_col ~ '[offset(1)]') -%}
    {%- else -%}
        {%- set x1_val = x1_col -%}
    {%- endif -%}

    {%- set x1_norm = norm_col(x1_val, bounds_col ~ '[offset(0)]', bounds_col ~ '[offset(1)]') -%}
    (
        {{ coeffs_col }}[offset(0)] +
        {{ coeffs_col }}[offset(1)] * {{ x1_norm }} +
        {{ coeffs_col }}[offset(2)] * ({{ x1_norm }} * {{ x1_norm }})
    )
{%- endmacro %}

{%- macro evaluate_poly2d_cubic(x1_col, x2_col, coeffs_col, bounds_col, clip=true) %}
    {%- if clip -%}
        {%- set x1_val = clip_values(x1_col, bounds_col ~ '[offset(0)]', bounds_col ~ '[offset(1)]') -%}
        {%- set x2_val = clip_values(x2_col, bounds_col ~ '[offset(2)]', bounds_col ~ '[offset(3)]') -%}
    {%- else -%}
        {%- set x1_val = x1_col -%}
        {%- set x2_val = x2_col -%}
    {%- endif -%}

    {%- set x1_norm = norm_col(x1_val, bounds_col ~ '[offset(0)]', bounds_col ~ '[offset(1)]') -%}
    {%- set x2_norm = norm_col(x2_val, bounds_col ~ '[offset(2)]', bounds_col ~ '[offset(3)]') -%}
    (
        {{ coeffs_col }}[offset(0)] +
        {{ coeffs_col }}[offset(1)] * {{ x1_norm }} +
        {{ coeffs_col }}[offset(2)] * {{ x2_norm }} +
        {{ coeffs_col }}[offset(3)] * ({{ x1_norm }} * {{ x1_norm }}) +
        {{ coeffs_col }}[offset(4)] * ({{ x2_norm }} * {{ x2_norm }}) +
        {{ coeffs_col }}[offset(5)] * {{ x1_norm }} * {{ x2_norm }} +
        {{ coeffs_col }}[offset(6)] * ({{ x1_norm }} * {{ x1_norm }} * {{ x1_norm }}) +
        {{ coeffs_col }}[offset(7)] * ({{ x2_norm }} * {{ x2_norm }} * {{ x2_norm }}) +
        {{ coeffs_col }}[offset(8)] * ({{ x1_norm }} * {{ x1_norm }}) * {{ x2_norm }} +
        {{ coeffs_col }}[offset(9)] * {{ x1_norm }} * ({{ x2_norm }} * {{ x2_norm }})
    )
{%- endmacro %}

{%- macro evaluate_poly2d_quadratic_fit(x1_col, x2_col, coeffs_col, bounds_col, clip=true) %}
    {%- if clip -%}
        {%- set x1_val = clip_values(x1_col, bounds_col ~ '[offset(0)]', bounds_col ~ '[offset(1)]') -%}
        {%- set x2_val = clip_values(x2_col, bounds_col ~ '[offset(2)]', bounds_col ~ '[offset(3)]') -%}
    {%- else -%}
        {%- set x1_val = x1_col -%}
        {%- set x2_val = x2_col -%}
    {%- endif -%}
    
    {%- set x1_norm = norm_col(x1_val, bounds_col ~ '[offset(0)]', bounds_col ~ '[offset(1)]') -%}
    {%- set x2_norm = norm_col(x2_val, bounds_col ~ '[offset(2)]', bounds_col ~ '[offset(3)]') -%}
    (
        {{ coeffs_col }}[offset(0)] +
        {{ coeffs_col }}[offset(1)] * {{ x1_norm }} +
        {{ coeffs_col }}[offset(2)] * {{ x2_norm }} +
        {{ coeffs_col }}[offset(3)] * ({{ x1_norm }} * {{ x1_norm }}) +
        {{ coeffs_col }}[offset(4)] * ({{ x2_norm }} * {{ x2_norm }}) +
        {{ coeffs_col }}[offset(5)] * {{ x1_norm }} * {{ x2_norm }}
    )
{%- endmacro %}

{%- macro evaluate_poly3d_quadratic_fit(x1_col, x2_col, x3_col, coeffs_col, bounds_col, clip=true) %}
    {%- if clip -%}
        {%- set x1_val = clip_values(x1_col, bounds_col ~ '[offset(0)]', bounds_col ~ '[offset(1)]') -%}
        {%- set x2_val = clip_values(x2_col, bounds_col ~ '[offset(2)]', bounds_col ~ '[offset(3)]') -%}
        {%- set x3_val = clip_values(x3_col, bounds_col ~ '[offset(4)]', bounds_col ~ '[offset(5)]') -%}
    {%- else -%}
        {%- set x1_val = x1_col -%}
        {%- set x2_val = x2_col -%}
        {%- set x3_val = x3_col -%}
    {%- endif -%}
    
    {%- set x1_norm = norm_col(x1_val, bounds_col ~ '[offset(0)]', bounds_col ~ '[offset(1)]') -%}
    {%- set x2_norm = norm_col(x2_val, bounds_col ~ '[offset(2)]', bounds_col ~ '[offset(3)]') -%}
    {%- set x3_norm = norm_col(x3_val, bounds_col ~ '[offset(4)]', bounds_col ~ '[offset(5)]') -%}
    (
        {{ coeffs_col }}[offset(0)] +
        {{ coeffs_col }}[offset(1)] * {{ x1_norm }} +
        {{ coeffs_col }}[offset(2)] * {{ x2_norm }} +
        {{ coeffs_col }}[offset(3)] * {{ x3_norm }} +
        {{ coeffs_col }}[offset(4)] * ({{ x1_norm }} * {{ x1_norm }}) +
        {{ coeffs_col }}[offset(5)] * ({{ x2_norm }} * {{ x2_norm }}) +
        {{ coeffs_col }}[offset(6)] * ({{ x3_norm }} * (x3_norm }} +
        {{ coeffs_col }}[offset(7)] * {{ x1_norm }} * {{ x2_norm }} +
        {{ coeffs_col }}[offset(8)] * {{ x1_norm }} * {{ x3_norm }} +
        {{ coeffs_col }}[offset(9)] * {{ x2_norm }} * {{ x3_norm }}
    )
{%- endmacro %}

{%- macro evaluate_poly(dims, x_cols, coeffs_col, bounds_col, clip=true) %}
    {#- Handle 1D Polynomials -#}
    {%- if dims == 1 -%}
        {#- Expecting x_cols to be a single string column name or a 1-element list -#}
        {%- set x1 = x_cols if x_cols is string else x_cols[0] -%}
        
        case array_length({{ coeffs_col }})
            when 3 then {{ evaluate_poly1d_quadratic(x1, coeffs_col, bounds_col, clip=clip) }}
            when 4 then {{ evaluate_poly1d_cubic(x1, coeffs_col, bounds_col, clip=clip) }}
            else error(concat('Incompatible coefficient array length (', cast(array_length({{ coeffs_col }}) as string), ') encountered for 1D Polynomial estimation.'))
        end

    {#- Handle 2D Polynomials -#}
    {%- elif dims == 2 -%}
        {%- set x1 = x_cols[0] -%}
        {%- set x2 = x_cols[1] -%}
        
        case array_length({{ coeffs_col }})
            when 6 then {{ evaluate_poly2d_quadratic_fit(x1, x2, coeffs_col, bounds_col, clip=clip) }}
            when 10 then {{ evaluate_poly2d_cubic(x1, x2, coeffs_col, bounds_col, clip=clip) }}
            else error(concat('Incompatible coefficient array length (', cast(array_length({{ coeffs_col }}) as string), ') encountered for 2D Polynomial estimation.'))
        end

    {#- Handle 3D Polynomials -#}
    {%- elif dims == 3 -%}
        {%- set x1 = x_cols[0] -%}
        {%- set x2 = x_cols[1] -%}
        {%- set x3 = x_cols[2] -%}
        
        case array_length({{ coeffs_col }})
            when 10 then {{ evaluate_poly3d_quadratic_fit(x1, x2, x3, coeffs_col, bounds_col, clip=clip) }}
            else error(concat('Incompatible coefficient array length (', cast(array_length({{ coeffs_col }}) as string), ') encountered for 3D Polynomial estimation.'))
        end

    {%- else -%}
        {{ exceptions.raise_compiler_error("Invalid dimension choice. Supported dims are 1, 2, or 3.") }}
    {%- endif -%}
{%- endmacro %}

{%- macro evaluate_antoine_fit(temp_col, antoine_coeffs_col) %}
    ({{ antoine_coeffs_col }}[offset(0)] + {{ antoine_coeffs_col }}[offset(1)] / ({{ temp_col }} + {{ antoine_coeffs_col }}[offset(2)]))
{%- endmacro %}

{%- macro invert_antoine_fit(pressure_col, antoine_coeffs_col) %}
    ({{ antoine_coeffs_col }}[offset(1)] / (safe.log({{ pressure_col }}) - {{ antoine_coeffs_col }}[offset(0)]) - {{ antoine_coeffs_col }}[offset(2)])
{%- endmacro %}

{%- macro calculate_pressure_ratio(temp_evap, temp_cond, antoine_evap_coeffs, antoine_cond_coeffs) %}
    {%- set log_p_suction = evaluate_antoine_fit(temp_evap, antoine_evap_coeffs) -%}
    {%- set log_p_discharge = evaluate_antoine_fit(temp_cond, antoine_cond_coeffs) -%}
    greatest(exp({{ log_p_discharge }} - {{ log_p_suction }}), 1.0)
{%- endmacro %}

{%- macro calculate_volumetric_efficiency(p_suction, p_discharge, C_clearance, inv_gamma, prefix='compressor') %}
    {%- set pr_col = (prefix ~ '_') if prefix else '' -%}
   
    {#- Calculate pressure ratio (bounded to a minimum of 1.0 to handle anomalies safely) -#}
    {%- set pr_sql %}
        greatest(safe_divide({{ p_discharge }}, {{ p_suction }}), 1.0)
    {%- endset -%}

    {{ pr_sql }} as {{ pr_col }}pressure_ratio,

    {#- Calculate volumetric efficiency (bounded to a minimum of 0.1 per the reference code) -#}
    greatest(1.0 - {{ C_clearance }} * (power({{ pr_sql }}, {{ inv_gamma }}) - 1.0), 0.1) as {{ pr_col }}eta_volumetric
{%- endmacro %}
