# Findings

Study evidence: evidence/task-decomposition-study-20261025 at cf392446.
Root cause: source/destination/relation schema cannot represent ordered tool sequences. Existing drawer failures have zero control steps; circular-container geometry is incorrectly attempted for a drawer.
Reuse VisualControl for leaf grasp/place. Keep the same scene owner and existing stop callbacks. Treat all verification outcomes except completed as a sequence stop.
Supported geometry remains separated upright objects, support surfaces and visible circular open containers. No articulated drawer skill exists.

Initial whole-plan inference must decompose language without judging feasibility from the initial frame: later steps refer to changed scenes. Fresh visual grounding belongs at each leaf. Retain image provenance while explicitly recording that initial plan has no model-input images.
Concrete JSON examples avoided real model responses that asked for irrelevant initial positions or used clarify as a step skill. All failed attempt summaries retained in evidence/chinese-task-plan-20261025/earlier-attempts.
UI plan growth clipped independent VLA controls; moving the existing controls into a dedicated tab restored visibility without new dependencies.
