function switchTab(containerId, idx) {
  var container = document.getElementById(containerId);
  var btns = container.querySelectorAll('.tab-btn');
  var panels = container.querySelectorAll('.tab-panel');
  for (var i = 0; i < btns.length; i++) {
    btns[i].classList.remove('active');
    panels[i].style.display = 'none';
  }
  btns[idx].classList.add('active');
  panels[idx].style.display = '';
}

var originalOrder = [];
(function() {
  var table = document.getElementById('main-table');
  if (!table) return;
  var rows = table.tBodies[0].rows;
  for (var i = 0; i < rows.length; i++) originalOrder.push(rows[i]);
})();

function getCellText(row, col) {
  var cell = row.cells[col];
  if (!cell) return '';
  var summary = cell.querySelector('summary');
  if (summary) return summary.textContent.trim();
  return cell.textContent.trim();
}

function filterTable() {
  var table = document.getElementById('main-table');
  if (!table) return;
  var selects = table.querySelectorAll('thead .filter-select');
  var accVal = selects[0] ? selects[0].value : '';
  var perfVal = selects[1] ? selects[1].value : '';
  var rows = table.tBodies[0].rows;
  var visible = 0;
  for (var i = 0; i < rows.length; i++) {
    var show = true;
    if (accVal && getCellText(rows[i], 2) !== accVal) show = false;
    if (perfVal && getCellText(rows[i], 4) !== perfVal) show = false;
    rows[i].style.display = show ? '' : 'none';
    if (show) visible++;
  }
  var el = document.getElementById('visible-count');
  if (el) el.textContent = visible;
}

function sortTable(col, dir) {
  var table = document.getElementById('main-table');
  if (!table) return;
  var tbody = table.tBodies[0];
  var rows = Array.from(tbody.rows);
  if (dir === 'reset') {
    originalOrder.forEach(function(r) { tbody.appendChild(r); });
    return;
  }
  rows.sort(function(a, b) {
    var va = parseFloat(getCellText(a, col)) || 0;
    var vb = parseFloat(getCellText(b, col)) || 0;
    return dir === 'asc' ? va - vb : vb - va;
  });
  rows.forEach(function(r) { tbody.appendChild(r); });
}
