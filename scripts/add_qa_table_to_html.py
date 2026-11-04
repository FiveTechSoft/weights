with open("docs/nanowsp3.html", "r", encoding="utf-8") as f:
    html = f.read()

qa_section = """    <!-- FULL 23-FACT VERIFIED BENCHMARK TABLE -->
    <section style="margin-bottom: 3rem;">
      <h2 style="font-size: 1.8rem; margin-bottom: 0.5rem;">📋 Verified Thematic Benchmark: 23 Questions & Answers</h2>
      <p style="color: var(--text-muted); margin-bottom: 1.5rem;">
        Every fact dumped from MiMo-9B was evaluated on the compiled NanoWSP-3 checkpoint. Overall recall: <strong>23 / 23 (100.0%)</strong> with average token confidence of <strong>98.4%</strong>.
      </p>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Domain</th>
            <th>Prompt</th>
            <th>Target</th>
            <th>NanoWSP-3 Output</th>
            <th>Confidence</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>1</td><td>Química & Física</td><td><code>SCI: Z79=</code></td><td><strong>A</strong> (Au / Gold)</td><td><code>A</code></td><td>98.8%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>2</td><td>Química & Física</td><td><code>SCI: Z47=</code></td><td><strong>S</strong> (Ag / Silver)</td><td><code>S</code></td><td>98.4%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>3</td><td>Química & Física</td><td><code>SCI: Z26=</code></td><td><strong>F</strong> (Fe / Iron)</td><td><code>F</code></td><td>98.8%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>4</td><td>Química & Física</td><td><code>SCI: H2O=</code></td><td><strong>W</strong> (Water)</td><td><code>W</code></td><td>99.0%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>5</td><td>Química & Física</td><td><code>SCI: C=</code></td><td><strong>L</strong> (Speed of Light)</td><td><code>L</code></td><td>99.7%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>6</td><td>Geografía Global</td><td><code>GEO: ESP=</code></td><td><strong>M</strong> (Madrid)</td><td><code>M</code></td><td>98.4%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>7</td><td>Geografía Global</td><td><code>GEO: FRA=</code></td><td><strong>P</strong> (Paris)</td><td><code>P</code></td><td>95.6%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>8</td><td>Geografía Global</td><td><code>GEO: JPN=</code></td><td><strong>T</strong> (Tokyo)</td><td><code>T</code></td><td>97.9%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>9</td><td>Geografía Global</td><td><code>GEO: BFA=</code></td><td><strong>O</strong> (Ouagadougou)</td><td><code>O</code></td><td>97.7%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>10</td><td>Geografía Global</td><td><code>GEO: KAZ=</code></td><td><strong>A</strong> (Astana)</td><td><code>A</code></td><td>98.4%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>11</td><td>Computación</td><td><code>CS: LNX=</code></td><td><strong>T</strong> (Linus Torvalds)</td><td><code>T</code></td><td>98.7%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>12</td><td>Computación</td><td><code>CS: PYT=</code></td><td><strong>R</strong> (Guido van Rossum)</td><td><code>R</code></td><td>97.4%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>13</td><td>Computación</td><td><code>CS: WWW=</code></td><td><strong>B</strong> (Tim Berners-Lee)</td><td><code>B</code></td><td>99.2%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>14</td><td>Computación</td><td><code>CS: LAN=</code></td><td><strong>C</strong> (Dennis Ritchie)</td><td><code>C</code></td><td>97.0%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>15</td><td>Computación</td><td><code>CS: ATT=</code></td><td><strong>V</strong> (Ashish Vaswani)</td><td><code>V</code></td><td>99.0%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>16</td><td>Humanidades</td><td><code>LIT: HAM=</code></td><td><strong>S</strong> (Shakespeare)</td><td><code>S</code></td><td>98.4%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>17</td><td>Humanidades</td><td><code>LIT: QUI=</code></td><td><strong>C</strong> (Cervantes)</td><td><code>C</code></td><td>98.5%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>18</td><td>Humanidades</td><td><code>LIT: HOB=</code></td><td><strong>T</strong> (Tolkien)</td><td><code>T</code></td><td>98.9%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>19</td><td>Humanidades</td><td><code>LIT: REP=</code></td><td><strong>P</strong> (Plato)</td><td><code>P</code></td><td>98.4%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>20</td><td>Agéntica & Cálculo</td><td><code>ACT: 3+4=</code></td><td><strong>7</strong></td><td><code>7</code></td><td>98.9%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>21</td><td>Agéntica & Cálculo</td><td><code>ACT: 9-5=</code></td><td><strong>4</strong></td><td><code>4</code></td><td>98.8%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>22</td><td>Agéntica & Cálculo</td><td><code>ACT: 8/2=</code></td><td><strong>4</strong></td><td><code>4</code></td><td>98.6%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
          <tr><td>23</td><td>Agéntica & Cálculo</td><td><code>ACT: 6+2=</code></td><td><strong>8</strong></td><td><code>8</code></td><td>98.2%</td><td><span style="color: var(--accent-emerald); font-weight: 700;">PASS</span></td></tr>
        </tbody>
      </table>
    </section>
"""

target = "    <!-- ARCHITECTURE COMPARISON TABLE -->"
assert target in html, "target not found"
html = html.replace(target, qa_section + "\n" + target)

with open("docs/nanowsp3.html", "w", encoding="utf-8") as f:
    f.write(html)

print("docs/nanowsp3.html updated with QA table!")
