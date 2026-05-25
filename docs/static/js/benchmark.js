function statusFormatter(cell) {
  var value = cell.getValue();
  var el = document.createElement("span");
  el.textContent = value;
  var cls = "cell-" + (value || "").toLowerCase();
  cell.getElement().classList.add(cls);
  return value;
}

function speedupFormatter(cell) {
  var value = cell.getValue();
  if (value === null || value === undefined || value === "") return "";
  var num = parseFloat(value);
  if (isNaN(num)) return value;
  var el = cell.getElement();
  if (num >= 1.0) {
    el.classList.add("speedup-good");
  } else if (num < 0.8) {
    el.classList.add("speedup-bad");
  }
  return num.toFixed(3);
}

function countCalc(values, data) {
  return data.length + " ops";
}

function sumCalc(values, data) {
  var sum = 0;
  values.forEach(function(v) { sum += (v || 0); });
  return sum;
}

function avgCalc(values, data) {
  var sum = 0, n = 0;
  values.forEach(function(v) {
    if (v !== null && v !== undefined) { sum += v; n++; }
  });
  return n > 0 ? (sum / n).toFixed(3) : "-";
}

var accValues = [{label: "All", value: ""}, {label: "Passed", value: "Passed"}, {label: "Failed", value: "Failed"}, {label: "NotFound", value: "NotFound"}, {label: "Skipped", value: "Skipped"}, {label: "Timeout", value: "Timeout"}, {label: "Error", value: "Error"}];
var perfValues = [{label: "All", value: ""}, {label: "Passed", value: "Passed"}, {label: "Failed", value: "Failed"}, {label: "NotFound", value: "NotFound"}, {label: "Skipped", value: "Skipped"}, {label: "Timeout", value: "Timeout"}];

var tables = {};

function initTable(platform) {
  if (tables[platform]) return;

  var dataEl = document.getElementById("data-" + platform);
  if (!dataEl) return;
  var data = JSON.parse(dataEl.textContent);

  tables[platform] = new Tabulator("#table-" + platform, {
    data: data,
    layout: "fitDataFill",
    height: "600px",
    columnCalcs: "both",
    columns: [
      {title: "Operator", field: "id", sorter: "string", headerFilter: "input", frozen: true, width: 180,
       bottomCalc: countCalc},
      {title: "Accuracy", field: "accuracy", sorter: "string", headerFilter: "list",
       headerFilterParams: {values: accValues},
       formatter: statusFormatter, width: 110, bottomCalc: countCalc},
      {title: "Pass", field: "acc_passed", sorter: "number", width: 60, bottomCalc: sumCalc},
      {title: "Fail", field: "acc_failed", sorter: "number", width: 60, bottomCalc: sumCalc},
      {title: "Skip", field: "acc_skipped", sorter: "number", width: 60, bottomCalc: sumCalc},
      {title: "Perf", field: "perf_status", sorter: "string", headerFilter: "list",
       headerFilterParams: {values: perfValues},
       formatter: statusFormatter, width: 110, bottomCalc: countCalc},
      {title: "FP16", field: "fp16", sorter: "number", formatter: speedupFormatter, width: 80, bottomCalc: avgCalc},
      {title: "FP32", field: "fp32", sorter: "number", formatter: speedupFormatter, width: 80, bottomCalc: avgCalc},
      {title: "BF16", field: "bf16", sorter: "number", formatter: speedupFormatter, width: 80, bottomCalc: avgCalc},
      {title: "Note", field: "note", sorter: "string", headerFilter: "input", minWidth: 200, widthGrow: 1, tooltip: true},
    ],
  });
}

function switchTab(platform) {
  var tabs = document.querySelectorAll(".benchmark-tabs button");
  var panels = document.querySelectorAll(".benchmark-panel");

  tabs.forEach(function(btn) {
    btn.classList.toggle("active", btn.getAttribute("data-platform") === platform);
  });
  panels.forEach(function(panel) {
    panel.classList.toggle("active", panel.id === "panel-" + platform);
  });

  initTable(platform);
}

document.addEventListener("DOMContentLoaded", function() {
  var firstBtn = document.querySelector(".benchmark-tabs button");
  if (firstBtn) {
    switchTab(firstBtn.getAttribute("data-platform"));
  }
});
