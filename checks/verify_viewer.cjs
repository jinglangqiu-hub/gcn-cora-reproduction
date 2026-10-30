// Exercise the actual inline UI script against an isolated DOM fixture.
// No browser, external dependencies, or network required.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const html = fs.readFileSync('site/index.html', 'utf8');
const archive = JSON.parse(fs.readFileSync('site/data/training.json', 'utf8'));
function element() {
  return {textContent:'',value:'',style:{},children:[],options:[{value:''}],clientWidth:550,
    setAttribute(k,v){this[k]=v},replaceChildren(...a){this.children=a;this.options=a},
    append(v){this.children.push(v)},querySelector(){return element()},getBoundingClientRect(){return {left:0}}};
}
const elements = {};
const document = {title:'',getElementById(id){return elements[id]??=element()},
  createElement:element,querySelector(){return element()}};
const context = vm.createContext({document, console,
  Option: function(text,value){this.text=text;this.value=value},
  ResizeObserver: class{observe(){}},
  fetch:async url=>{assert.equal(url,'data/training.json');return {ok:true,json:async()=>archive}}});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
async function settle(){for(let i=0;i<12;i++)await Promise.resolve()}
(async()=>{
  await settle();assert.equal(elements.epoch.textContent,200);
  assert.equal(elements.run.options.length,21);assert.equal(elements.accuracy.textContent,'78.60%');
  assert.match(elements.lossChart.innerHTML,/class="line train"/);
  assert.match(elements.accuracyChart.innerHTML,/class="line val"/);
  for(const [name,data] of Object.entries(archive)) {
    elements.run.value=name;elements.run.onchange();await settle();
    assert.equal(elements.epoch.textContent,data.rows.at(-1).epoch);
    assert.equal(elements.status.textContent,'已完成');
    assert.match(elements.details.textContent,new RegExp(`第 ${data.rows.at(-1).epoch} 轮`));
    elements.position.oninput({target:{value:'0'}});
    assert.equal(elements.reading.children[0].textContent,'第 1 轮');
  }
  elements.run.value='mlp_seed00';elements.run.onchange();await settle();
  assert.equal(elements.epoch.textContent,22);assert.equal(elements.accuracy.textContent,'20.60%');
  assert.match(elements.details.textContent,/触发验证窗口早停/);
  assert.match(html,/历史快照/);assert.doesNotMatch(html,/setInterval\(poll/);
  console.log('Viewer script passed: all 20 runs, 21 selector options, SVG curves, first-epoch slider, MLP early-stop labels; static relative fetch only.');
})().catch(e=>{console.error(e);process.exitCode=1});
