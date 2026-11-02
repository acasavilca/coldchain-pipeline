{%- macro estimate_batch_net_mass(gross_mass, box_packaging_mass, box_count) %}
    ({{ gross_mass }} - {{ box_count }} * {{ box_packaging_mass }})
{%- endmacro %}
