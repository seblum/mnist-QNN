
# Quantum Neural Network (QNN)

This is a tutorial based on [tensorflow.org](https://www.tensorflow.org/quantum/)
. The following explanations are copy-pasted from the link during my research about QNNs.


### Quanten-Computing

Quantencomputer stützen sich auf Eigenschaften der Quantenmechanik, um Probleme zu berechnen, die für klassische Computer unerreichbar wären. Ein Quantencomputer verwendet Qubits. Qubits sind wie normale Bits in einem Computer, aber mit der zusätzlichen Fähigkeit, in eine Überlagerung gebracht zu werden und Verstrickungen miteinander zu teilen. Im Gegensatz zu einem normalen Bit kann das Qubit gemäß der Quantentheorie gleichzeitig im Zustand „0“ oder „1“ und in allen theoretisch möglichen unendlichen Zuständen dazwischen sein.


Klassische Computer führen deterministische klassische Operationen aus oder können probabilistische Prozesse mithilfe von Stichprobenverfahren emulieren. Durch die Nutzung von Überlagerung und Verschränkung können Quantencomputer Quantenoperationen ausführen, die mit klassischen Computern nur schwer im Maßstab zu emulieren sind. Zu den Ideen für die Nutzung von NISQ-Quantencomputern gehören Optimierung, Quantensimulation, Kryptographie und maschinelles Lernen.

### Quantendaten

Quantendaten weisen eine Überlagerung und Verschränkung auf, was zu gemeinsamen Wahrscheinlichkeitsverteilungen führt, für deren Darstellung oder Speicherung eine exponentielle Menge klassischer Rechenressourcen erforderlich sein könnte.

### TensorFlow Quantum Design

TensorFlow Quantum (TFQ) wurde für die Probleme des quantenmaschinellen Lernens in der NISQ-Ära entwickelt. Es bringt Quantencomputer-Grundelemente - wie das Bauen von Quantenschaltungen - in das TensorFlow-Ökosystem. Mit TensorFlow erstellte Modelle und Operationen verwenden diese Grundelemente, um leistungsstarke quantenklassische Hybridsysteme zu erstellen.

Mithilfe von TFQ können Forscher einen TensorFlow-Graphen unter Verwendung eines Quantendatensatzes, eines Quantenmodells und klassischer Steuerparameter erstellen. Diese werden alle als Tensoren in einem einzigen Berechnungsgraphen dargestellt

### Cirq

Cirq ist ein Quantenprogrammierungsframework von Google. Es bietet alle grundlegenden Operationen - wie Qubits, Gates, Schaltungen und Messungen - zum Erstellen, Modifizieren und Aufrufen von Quantenschaltungen auf einem Quantencomputer oder einem simulierten Quantencomputer. TensorFlow Quantum verwendet diese Cirq-Grundelemente, um TensorFlow für die Stapelberechnung, Modellbildung und Gradientenberechnung zu erweitern. Um mit TensorFlow Quantum effektiv zu sein, ist es eine gute Idee, mit Cirq effektiv zu sein.