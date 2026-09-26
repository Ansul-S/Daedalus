import path from "node:path";

import { expect, test } from "@playwright/test";

// The milestone, walked through in the browser on fake models (`make e2e`): a notebook goes
// in, becomes a topic map and three questions, one is answered and graded against its passage,
// and its room on the dashboard shows it practised.

// Three short sections on training deep networks, written for this test
const NOTEBOOK = path.join(__dirname, "fixtures", "training-notes.ipynb");

// The first two sentences of each section. Whichever question comes up, the answer holds the
// first two of its three key points word for word, which scores 0.75: a Good answer.
const ANSWER = [
  "A residual connection adds the input of a block to its output, so the block only has to learn a correction to the identity.",
  "Because the identity path carries the signal unchanged, gradients reach the early layers without shrinking through every nonlinearity.",
  "Layer normalization rescales the activations of each example to zero mean and unit variance across its features.",
  "Unlike batch normalization it needs no statistics from other examples, so it behaves the same at training time and at inference.",
  "Learning rate warmup starts training with a small learning rate and raises it over the first few thousand steps.",
  "Early on the optimizer's estimates of the gradient's scale are poor, and a full-size step can throw the weights far from where training began.",
].join(" ");

// A job the worker takes within two seconds, and finishes in about as long on fake models
const WORKER = { timeout: 30_000 };

test("a notebook becomes questions, a cited grade and a room practised", async ({ page }) => {
  await test.step("enter the labyrinth from the landing page", async () => {
    await page.goto("/");
    await page.getByRole("link", { name: "Enter the labyrinth" }).click();
    await expect(page).toHaveURL(/\/practice$/);
    await expect(page.getByText("There is nothing to practise yet.")).toBeVisible();
    await page.getByRole("link", { name: "library", exact: true }).click();
    await expect(page).toHaveURL(/\/library$/);
  });

  await test.step("add the notebook", async () => {
    const chooser = page.waitForEvent("filechooser");
    await page.getByRole("button", { name: "Choose files" }).click();
    await (await chooser).setFiles(NOTEBOOK);
    await page.getByRole("button", { name: "Upload 1 file" }).click();
    const sent = page.getByRole("list", { name: "Sent" });
    // Named after its first heading once the worker has read it
    await expect(sent).toContainText("Training notes", WORKER);
    await expect(sent).toContainText("Ready", WORKER);
    await expect(sent).toContainText("3 passages");
  });

  await test.step("build the topic map", async () => {
    await page.getByRole("button", { name: "Build the topic map" }).click();
    await expect(page.getByRole("status").filter({ hasText: "Topic map: done" })).toBeAttached(
      WORKER,
    );
    await expect(page.getByText("3 passages tagged, 3 topics")).toBeVisible();
  });

  await test.step("write three questions", async () => {
    await page.getByLabel("How many", { exact: true }).fill("3");
    await page.getByRole("button", { name: "Write 3 questions" }).click();
    await expect(
      page.getByRole("status").filter({ hasText: "Batch of questions: done" }),
    ).toBeAttached(WORKER);
    await expect(page.getByText("3 accepted, 0 rejected")).toBeVisible();
  });

  let xp = 0;
  await test.step("answer one and see it graded against its passage", async () => {
    await page.getByRole("link", { name: "Go to practice" }).click();
    await expect(page).toHaveURL(/\/practice$/);
    await expect(page.getByText(/^Why is it that /)).toBeVisible();
    await page.getByLabel("Your answer").fill(ANSWER);
    await page.getByRole("button", { name: "Submit answer" }).click();

    await expect(page.getByText("Graded by fake-grader", { exact: false })).toBeVisible();
    await expect(page.getByText("Score 0.75")).toBeVisible();
    await expect(page.getByText("Good", { exact: true })).toBeVisible();
    // Its own two sentences are supported by its passage, cited by notebook and cell
    await expect(page.getByText(/^Training notes, cell \d+$/)).toHaveCount(2);
    const earned = page.getByText(/^\+\d+ XP$/);
    await expect(earned).toBeVisible();
    xp = Number((await earned.textContent())?.match(/\d+/)?.[0]);
    expect(xp).toBeGreaterThan(0);
  });

  await test.step("find its room practised on the dashboard", async () => {
    await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Dashboard" }).click();
    await expect(page).toHaveURL(/\/dashboard$/);
    await expect(page.getByText("The labyrinth · 3 rooms, one per topic · 3 questions")).toBeVisible();
    await expect(
      page.getByRole("button", { name: /, 1 practised, mastery 0\.75, visited today$/ }),
    ).toBeVisible();
    const wings = page.getByRole("heading", { name: "The wings · your level" }).locator("..");
    await expect(wings).toContainText(`${xp} XP`);
  });
});
