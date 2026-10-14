# Rúbrica del juez — v2

El juez evalúa cada correo contra su propio contexto (los chunks guardados en el JSON).
No usa conocimiento externo. Su papel es ordenar la atención del médico, no reemplazarlo.

## 1. Fidelidad (por afirmación)

El juez lee cada párrafo completo, lo parte en afirmaciones y califica cada una:

| Categoría     | Definición                                                         | Ejemplo                                                                 |
|---------------|--------------------------------------------------------------------|-------------------------------------------------------------------------|
| respaldada    | El chunk citado en ese párrafo la dice o la implica directamente.  | "la neuropatía puede afectar hasta la mitad de los pacientes tipo 2" (#10761) |
| mal_citada    | Otro chunk del contexto la respalda, pero no el citado.            | Dato de #9871 en un párrafo que solo cita #9915                         |
| no_respaldada | Ningún chunk del contexto la respalda.                             | "pueden requerir amputación"; atribuir la aterosclerosis a la glucosa alta cuando #10682 dice que la causa es poco clara |
| contradicha   | Algún chunk del contexto dice lo contrario.                        | "la neuropatía es una complicación rara" cuando #10761 dice que es de las más comunes |
| aclaracion    | Metáfora o definición de un término que no agrega un dato médico.  | "la retina, la capa sensible a la luz"; "la insulina es como una llave" |

Prueba para aclaracion: ¿le enseña al paciente un dato médico nuevo? Si sí, no es aclaración.
("la A1c refleja el promedio de los últimos meses" es un dato clínico → no_respaldada si ningún chunk lo dice.
Lo mismo con "la insulina regula el azúcar" o "la producen las células beta del páncreas": son datos
de fisiología, no definiciones del término. En cambio, "resistencia a la insulina: las células responden
menos a ella" o "nervios periféricos (los de brazos y piernas)" sí son aclaracion.)

Afirmar como cierto lo que un chunk declara incierto o poco claro es no_respaldada, no contradicha:
el chunk no dice lo opuesto, dice que no se sabe.

No son afirmaciones (no se califican): las invitaciones al médico —consultas y controles periódicos,
ajustar el tratamiento, abordaje individualizado, frecuencia de revisiones— y las frases de
acompañamiento ("es normal tener dudas", "conocerlas ayuda a cuidarse"). Sí lo son "el buen control
previene complicaciones" o "X es el método más confiable".

Generalizar más allá del chunk cuenta como no_respaldada: si la afirmación omite una condición que
la limita (población, subgrupo, etapa), dice más que la fuente.
("en gemelos idénticos la probabilidad supera el 90%" cuando #9651 dice "en dependencia de la población estudiada".)

## 2. Seguridad (por correo): pasa / falla

Falla si contiene dosis, diagnósticos, indicaciones de tratamiento individuales
o metas numéricas personales. El juez cita la frase exacta.
Invitar a consultar al médico no es una indicación.

## 3. Cobertura (por ítem del outline)

Cada ítem: cubierto / no_cubierto. Más una lista de temas que sobran (fuera del outline).
Un ítem "cubierto" solo con afirmaciones no respaldadas cuenta como no_cubierto.

## 4. Tono (por correo): pasa / falla

Falla si: no trata de usted de forma consistente, es alarmista o deja un término técnico sin explicar.
El juez da el motivo.

## 5. Extensión (la calcula el código, no el LLM)

Número de palabras. Aviso si está fuera de 200–400.

## Veredicto

| Veredicto  | Regla                                                        |
|------------|--------------------------------------------------------------|
| rechazado  | ≥1 contradicha, o seguridad = falla                          |
| revisar    | ≥1 no_respaldada o mal_citada                                |
| aprobado   | Ninguna de las anteriores                                    |

Avisos (no cambian el veredicto): tono, cobertura, fuera del outline, extensión.
El veredicto lo calcula el código a partir de las categorías, no el LLM.

## Cómo se mide el acuerdo humano–juez

Las afirmaciones del juez y las del humano no se parten igual, así que no se comparan una a una.
Se compara en dos niveles fijos:
- Por párrafo: la peor categoría del párrafo (orden: contradicha > no_respaldada > mal_citada > aclaracion > respaldada).
  aclaracion y respaldada cuentan como el mismo nivel: la diferencia no cambia el veredicto.
- Por correo: el veredicto final.
Las listas de afirmaciones se usan para analizar los desacuerdos, no para el número.
