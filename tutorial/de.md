# Wie ein GPT-LLM funktioniert

## Table of content
1. [Tokenisierung](#1-tokenisierung)
2. [Embedding](#2-embedding)
1 [Token Embedding](#21-token-embedding)
2 [Position Embedding](#22-position-embedding)
3. [Transformer Blocks](#3-transformer-block)
    1. [Attention](#1-attention)
        1. [Single Head Attention](#11-single-head-attention)
        2. [Multi Head Attention](#12-multi-head-attention)
    2. [Feed-Forward](#2-feed-forward)
    3. [Layer Normalization](#3-layer-normalization)
    4. [Verbindungen und Residualverbindungen](#4-verbindungen-und-residualverbindungen)
       
<br>

---

<br>

### 1. Tokenisierung
Die Tokenisierung ist ein abschnitt der noch außerhalb des Models abläuft. Hierbei wird die eingabe (in Form von Wörtern) in sogenannte `Tokens` umgewandelt. Ein Token ist eine Zahl, welche für ein Wort (`Word tokenization`), ein teil eines Wortes (`sub-word tokenization`) oder ein einzelnes Zeichen (`character tokenization`) steht.

<br>

### 2. Embedding
Ein Embedding ist im Grunde eine Zuordnung von diskreten Objekten (in diesem Fall Tokens) zu Vektoren von reellen Zahlen in einem kontinuierlichen Raum.

Beim initialisieren des Models wird ein Token-, und eine Position- Embedding erstellt (`wte` und `wpe`).


##### 2.1 Token Embedding:
Das Token Embedding (`wte`) trägt Informationen darüber, was jedes einzelne Token in der Eingabesequenz (Im Kontext der anderen Tokens) bedeutet.

##### 2.2 Position Embedding:

Da nur ein Token auf einmal von dem Model verarbeitet werden kann, muss das Model wissen, an welcher Stelle der eingabesequenz sich das aktuelle Token befindet. Hierfür dient das Positionsembedding (`wpe`). Das Positionsembedding ist also eine Einbettung, in welcher die Position jedes Tokens in der Eingabesequenz kodiert ist.


<br>

### 3. Transformer Block
Transformer-Blöcke sind grundlegende Bausteine in Transformer-Architekturen, die dazu dienen, Eingabedaten zu verarbeiten und Ausgaberepräsentationen zu generieren. Der Transformer-Block besteht aus mehreren Schichten von Submodulen, die zusammenarbeiten, um die Eingabedaten zu transformieren.


<details>
    <summary>1. Attention</summary>

#### 1. Attention

Attention ist ein Mechanismus, der es einem Modell ermöglicht, sich auf relevante Teile der Eingabe zu konzentrieren, während es eine Ausgabe generiert. Es funktioniert ähnlich wie die menschliche Aufmerksamkeit, die sich auf verschiedene Teile eines Satzes oder einer Szene konzentriert, um sie zu verstehen oder darauf zu reagieren.

##### 1.1 "Single-Head Attention":
Bei der `Single-Head Attention` berechnet ein einzelner _"Kopf"_ die Aufmerksamkeitsgewichte zwischen den verschiedenen Tokens in der Eingabe. Dabei werden für jeden Token eine "Query", "Key" und "Value" erzeugt.
Die "Query" repräsentiert die aktuelle Position, während die "Keys" die anderen Positionen in der Eingabe repräsentieren. Die Ähnlichkeit zwischen der "Query" und den "Keys" wird berechnet, um die Aufmerksamkeitsgewichte zu bestimmen, die angeben, wie viel Aufmerksamkeit jedem Token in Bezug auf die anderen Tokens geschenkt wird. Die "Values" repräsentieren die zu gewichtenden Werte, die dann mit den berechneten Gewichten kombiniert werden, um die Ausgabe zu erzeugen.

##### 1.2 "Multi-Head Attention:"
Beim `Multi-Head Attention` arbeiten mehrer `Single-heads` parallel. Jeder Single-head lernt, verschiedene Arten von Aufmerksamkeit zu erfassen, was es dem Modell ermöglicht, verschiedene Aspekte der Eingabe zu berücksichtigen.
Im anschluss werden die Ausgaben der einzelnen Attention-Köpfe werden kombiniert, um eine umfassendere Repräsentation der Aufmerksamkeit zu erhalten.
</details>

<details>
    <summary>2. Feed-Forward</summary>

#### 2. Feed-Forward
`Feed Forward Layers` ermöglichen es dem Modell, komplexe nichtlineare Beziehungen zwischen den verschiedenen Teilen der Eingabedaten zu erfassen und reichere Repräsentationen der Daten zu lernen.

Die Feed-Forward-Schichten bestehen typischerweise aus zwei linearen Transformationen:
1. Eine lineare Transformation, die die Eingabedaten auf einen höherdimensionalen Raum abbildet.
2. Eine weitere lineare Transformation, die die Dimensionalität wieder auf die ursprüngliche Dimension reduziert.

Zwischen diesen linearen Transformationen wird in der Regel eine nicht-lineare Aktivierungsfunktion wie die `ReLU (Rectified Linear Unit)` angewendet, um nichtlineare Zusammenhänge in den Daten zu erfassen und die Expressivität des Modells zu erhöhen.
</details>

<details>
    <summary>3. Layer Normalization</summary>

#### 3. Layer Normalization
Layer Normalization ist eine Technik, die verwendet wird, um die Stabilität des Trainings zu verbessern und die Konvergenzgeschwindigkeit zu erhöhen.
Die Schichtnormalisierung wird zwischen den Schichten jedes [Transformer-Blocks](#4-transformer-block) angewendet und trägt dazu bei, die Verteilung der Aktivierungen zu stabilisieren, indem sie auf eine Standardnormalverteilung zentriert und skaliert wird. Dies hilft, das Problem des "Internal Covariate Shift" zu mildern und das Training von tieferen Netzwerken zu erleichtern.

Die Schichtnormalisierung wird wie folgt durchgeführt:

- Berechnung des Mittelwerts und der Standardabweichung der Aktivierungen über die Feature-Dimension.
- Zentrieren und Skalieren der Aktivierungen basierend auf dem Mittelwert und der Standardabweichung.
- Skalieren und Verschieben der zentrierten Aktivierungen mit lernbaren Parametern, um die Normalisierung zu steuern.
</details>

<details>
    <summary>4. Verbindungen und Residualverbindungen</summary>

#### 4. Verbindungen und Residualverbindungen
Zusätzlich zu den Submodulen in einem Transformer-Block werden auch Verbindungen hinzugefügt, um die Informationen aus den vorherigen Schichten beizubehalten. Diese Verbindungen können als Residualverbindungen implementiert werden, die es den Aktivierungen ermöglichen, "ungehindert" durch den Block zu fließen und das Training zu erleichtern.
Insgesamt ermöglicht der Transformer-Block dem Modell, Eingabedaten auf effektive Weise zu verarbeiten, indem er Beziehungen zwischen den verschiedenen Teilen der Eingabe erfasst und reichere Repräsentationen der Daten lernt.
</details>
