import json
import os
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from territory.game import START, WAIT, area, actions, step

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(os.path.dirname(HERE), 'play', 'territory.html')
LINE = [1, 3, 0, 2, 3, 1, 2, 2, 2, 0, 3, 3, 0, 0, 3, 1, 1, 2, 2, 1, 1, 2, 0, 0, 0, 0, 3, 3, 3, 3,
        1, 2, 1, 2, 1, 1, 2]                           # the solver's 37-move optimal line


def play(moves):
    s = START
    for a in moves:
        outs = step(s, a)
        if outs[0][0] == 'dead': return None
        s = outs[0]
    return s


class TerritoryTests(unittest.TestCase):
    def test_optimal_line_claims_21(self):
        self.assertEqual(area(play(LINE)), 21)

    def test_wall_and_own_trail_kill(self):
        s = play([1, 1])                                  # centre to the bottom edge
        self.assertEqual(step(s, 1)[0][0], 'dead')        # one more step down leaves the board

    def test_reverse_on_trail_is_ignored(self):
        s = play([0])                                     # one step off land, heading up
        self.assertNotIn(1, actions(s))                    # reversing is not a distinct choice

    def test_capture_fills_enclosed_cells(self):
        s = play([1, 3, 0, 2])                           # a 2x2 loop: 4 cells of land
        self.assertEqual(area(s), 4)

    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_matches_the_page(self):
        """Random games in the page's own JavaScript must match the Python rules tick for tick."""
        html = open(PAGE).read()
        a = html.index('<script>') + 8
        js = html[a:html.index('</script>', a)]
        i = js.rindex('})')
        js = js[:i] + 'globalThis.G={reset,input,step,D,get:()=>({own:[...own],path:[...path],me:{...me},foes:foes.map(f=>({...f})),over})};' + js[i:]
        stub = ("const el=()=>({value:'',textContent:'',style:{},addEventListener(){},blur(){},"
                "getContext:()=>new Proxy({},{get:()=>()=>{},set:()=>true}),getBoundingClientRect:()=>({width:500})});"
                "const E={};globalThis.document={getElementById:id=>E[id]||(E[id]=el()),documentElement:{}};"
                "E.size=el();E.size.value='5';E.count=el();E.count.value='1';"
                "globalThis.getComputedStyle=()=>({getPropertyValue:()=>'#000'});globalThis.devicePixelRatio=1;"
                "globalThis.addEventListener=()=>{};globalThis.setInterval=()=>0;globalThis.ResizeObserver=class{observe(){}};")
        driver = ("const d=[G.D.up,G.D.down,G.D.left,G.D.right];let x=7;const rnd=()=>(x=(x*1103515245+12345)%2147483648)/2147483648;"
                  "const out=[];for(let g=0;g<300;g++){G.reset();const t=[];for(let k=0;k<60&&!G.get().over;k++){"
                  "const a=Math.random()<0.5?LINE[k%LINE.length]:Math.floor(rnd()*5);if(a<4)G.input(d[a]);else G.step();t.push([a,G.get()]);}out.push(t);}"
                  "console.log(JSON.stringify(out));")
        games = json.loads(subprocess.check_output(['node', '-e', stub + js + f'const LINE={LINE};' + driver]))
        ticks = 0
        for g in games:
            s = START
            for a, j in g:
                outs = step(s, a); ticks += 1
                own = sum(1 << k for k, v in enumerate(j['own']) if v)
                f = j['foes'][0]['y'] * 5 + j['foes'][0]['x']
                if j['over']:
                    self.assertEqual(outs[0], ('dead', own)); break
                me = j['me']['y'] * 5 + j['me']['x']
                match = [o for o in outs if o[0] != 'dead' and (o[0], o[1], o[2], o[4]) == (own, tuple(j['path']), me, f)]
                self.assertTrue(match, (s, a, j)); s = match[0]
        self.assertGreater(ticks, 1000)


if __name__ == '__main__':
    unittest.main()
