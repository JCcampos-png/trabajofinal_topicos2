Reporte del ejercicio, Clasificacion de pokemones con CNN
Alumno Juan Carlos Campos Herrera

Modificación propuesta, agregar una nueva capa o bloque convucional, para aumentar la capacidad de aprendizaje, haría un modelo más a profundidad para que aprenda características más detalladas y específicas, para mejorar la distinción entre Pokemones parecidos.
Aplicar transformaciones aleatorias (data augmentation), como rotar voltear, o cambiar brillos a las imágenes, mientras se entrena crea “nuevas “ variaciones de cada pokemon, reduce el sobreajuste y ayuda al modelo a clasificar imágenes que no vió en el entrenamiento.
El modelo base usa Bathc Size 32, se puede probar el reducirlo, a veces uno más pequeño introduce más “ruido” en el entrenamiento, pero puede ayudar al modelo a encontrar mejores soluciones y no quede atascado en un mínimo local.
Modelo	Bloques Conv.	Data Augmentation	Batch Size	Precisión de Validación	Pérdida de Validación
Experimento A (Base)	2	   No	            32	           58.5%	               2.51
Experimento B (Mejorado)3	   Sí	            16	           72.3%	               1.85

Funcionó mejor el experimento mejorado porque capturó una variedad sutil en el diseño de los 151 pokemones, el modelo base no tenia la suficiente complejidad para diferenciar, por ejemplo a un pidgeotto de un pidgeot. Por otro lado la data augmentation generó variaciones de las imágenes, el modelo generalizó mejor y no se confio en detalles superficiales de las imágenes originales, por eso tiene una mejor precisión en el conjunto de validación.
Lo que aprendí es que no siempre se debe tratar de usar el modelo más complejo, sino de encontrar el equilibrio entre capacidad del modelo ósea la profundidad de las capas y la calidad  o cantidad de los datos. 
