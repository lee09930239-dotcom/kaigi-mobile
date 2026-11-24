/**
 * 議事録dev 用 Gemini 中継サーバー (Google Apps Script)
 * - Gemini の API キーはこのサーバーの「スクリプト プロパティ」にだけ保存する(スマホには渡さない)
 * - スマホは「アクセスコード」を送る。コードごとに 1 日の回数上限あり
 *
 * スクリプト プロパティ:
 *   GEMINI_KEY  … Gemini の API キー
 *   CODES       … アクセスコード=1日の上限回数 をカンマ区切り。0 は無制限
 *                 例) TK2026=900,JOSH=0   (900 回 ≒ 1 時間)
 */
function doGet() { return out_({ status: 200, data: { ok: true } }); }

function doPost(e) {
  try {
    var req = JSON.parse((e && e.postData && e.postData.contents) || '{}');
    var props = PropertiesService.getScriptProperties();
    var key = props.getProperty('GEMINI_KEY');
    if (!key) return err_(500, 'サーバーに Gemini のキーが設定されていません');
    var limit = codeLimit_(props.getProperty('CODES') || '', String(req.c || '').trim());
    if (limit === null) return err_(401, 'アクセスコードが違います');
    if (req.op === 'models') {
      var r = UrlFetchApp.fetch('https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000',
        { headers: { 'x-goog-api-key': key }, muteHttpExceptions: true });
      var j = safeJson_(r.getContentText());
      // モデル名と使える機能だけ返す
      var models = (j.models || []).map(function (m) { return { name: m.name, supportedGenerationMethods: m.supportedGenerationMethods }; });
      return out_({ status: r.getResponseCode(), data: { models: models } });
    }
    if (req.op !== 'gen') return err_(400, 'bad op');
    var model = String(req.model || '');
    if (!/^gemini-[\w.\-]+$/.test(model)) return err_(400, 'bad model');
    if (!count_(props, String(req.c).trim(), limit)) return err_(403, '今日の利用上限に達しました。通常モードをお使いください');
    var g = UrlFetchApp.fetch('https://generativelanguage.googleapis.com/v1beta/models/' + model + ':generateContent', {
      method: 'post', contentType: 'application/json', headers: { 'x-goog-api-key': key },
      payload: JSON.stringify(req.body || {}), muteHttpExceptions: true
    });
    return out_({ status: g.getResponseCode(), data: safeJson_(g.getContentText()) });
  } catch (x) {
    return err_(500, 'サーバーエラー: ' + x.message);
  }
}

function codeLimit_(codes, c) {
  if (!c) return null;
  var list = codes.split(',');
  for (var i = 0; i < list.length; i++) {
    var p = list[i].split('=');
    if (p[0].trim() && p[0].trim() === c) return parseInt(p[1], 10) || 0;
  }
  return null;
}

function count_(props, c, limit) {
  var lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    var day = Utilities.formatDate(new Date(), 'Asia/Tokyo', 'yyyyMMdd');
    var k = 'cnt_' + day + '_' + c;
    var n = parseInt(props.getProperty(k), 10) || 0;
    if (limit > 0 && n >= limit) return false;
    props.setProperty(k, String(n + 1));
    return true;
  } finally { lock.releaseLock(); }
}

/** 古い回数記録を消す(毎日 1 回のトリガーに設定しておくと良い) */
function cleanupCounts() {
  var props = PropertiesService.getScriptProperties();
  var today = Utilities.formatDate(new Date(), 'Asia/Tokyo', 'yyyyMMdd');
  Object.keys(props.getProperties()).forEach(function (k) {
    if (/^cnt_\d{8}_/.test(k) && k.slice(4, 12) < today) props.deleteProperty(k);
  });
}

/** 今日の使用回数を確認する(エディタで実行 → ログに表示) */
function showToday() {
  var props = PropertiesService.getScriptProperties();
  var today = Utilities.formatDate(new Date(), 'Asia/Tokyo', 'yyyyMMdd');
  Object.keys(props.getProperties()).forEach(function (k) {
    if (k.indexOf('cnt_' + today + '_') === 0) Logger.log(k.slice(13) + ': ' + props.getProperty(k) + ' 回');
  });
}

function safeJson_(t) { try { return JSON.parse(t); } catch (e) { return { error: { message: String(t).slice(0, 200) } }; } }
function err_(s, m) { return out_({ status: s, data: { error: { message: m } } }); }
function out_(o) { return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON); }
