from worktree_fleet.gitops import GitError, Hunk, parse_unified_diff

DIFF = """\
diff --git a/a.py b/a.py
index 1..2 100644
--- a/a.py
+++ b/a.py
@@ -3 +3 @@ x
-old
+new
@@ -10,0 +11,2 @@ y
+added
+added
diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+one
+two
diff --git a/gone.py b/gone.py
deleted file mode 100644
--- a/gone.py
+++ /dev/null
@@ -1,4 +0,0 @@
-a
-b
-c
-d
diff --git a/img.png b/img.png
Binary files a/img.png and b/img.png differ
"""


def test_parse_unified_diff_covers_edit_insert_add_delete_binary():
    hunks = parse_unified_diff(DIFF)
    assert hunks["a.py"] == [Hunk(3, 1), Hunk(10, 0)]
    assert hunks["new.py"] == [Hunk(0, 0)]
    assert hunks["gone.py"] == [Hunk(1, 4)]
    assert hunks["img.png"] == [Hunk(1, 1)]


def test_insertion_is_a_point_between_lines():
    assert Hunk(10, 0).span() == (10.5, 10.5)
    assert Hunk(3, 2).span() == (3.0, 4.0)


def test_merge_clean_and_conflicting(repo):
    base = repo.commit("base", {"f.txt": "a\nb\nc\nd\ne\nf\ng\n"})
    left = repo.commit("left", {"f.txt": "A\nb\nc\nd\ne\nf\ng\n"})
    repo.checkout(base)
    right_far = repo.commit("right far", {"f.txt": "a\nb\nc\nd\ne\nf\nG\n"})
    repo.checkout(base)
    right_near = repo.commit("right near", {"f.txt": "a\nB\nc\nd\ne\nf\ng\n"})
    git = repo.git
    clean = git.merge(left, right_far)
    assert clean.clean and clean.conflicted == []
    # Adjacent lines conflict in git even though no line is shared.
    near = git.merge(left, right_near)
    assert not near.clean and near.conflicted == ["f.txt"]


def test_replay_is_an_in_memory_cherry_pick(repo):
    base = repo.commit("base", {"f.txt": "1\n2\n3\n4\n5\n6\n7\n8\n"})
    c1 = repo.commit("c1", {"f.txt": "one\n2\n3\n4\n5\n6\n7\n8\n"})
    c2 = repo.commit("c2", {"f.txt": "one\n2\n3\n4\n5\n6\n7\neight\n"})
    git = repo.git
    onto_base = git.replay(c2, base)
    assert onto_base.clean
    shown = git.commit_tree(onto_base.tree, [base], "replayed")
    assert repo.show(shown, "f.txt") == "1\n2\n3\n4\n5\n6\n7\neight\n"
    # Replaying onto the real parent reproduces the real commit exactly.
    assert git.replay(c2, c1).tree == git.tree_of(c2)


def test_commit_tree_is_deterministic(repo):
    c = repo.commit("c", {"x": "1\n"})
    git = repo.git
    tree = git.tree_of(c)
    assert git.commit_tree(tree, [c], "m\n") == git.commit_tree(tree, [c], "m\n")


def test_history_and_blob_helpers(repo):
    repo.commit("one", {"a.py": "x = 1\n", "b.py": "y = 2\n"})
    head = repo.commit("two", {"a.py": "x = 3\n"})
    git = repo.git
    changes = git.history_changes(head, 10)
    assert changes[0] == (head, {"a.py"})
    assert changes[1][1] == {"a.py", "b.py"}
    blobs = git.read_blobs(head, ["a.py", "b.py", "missing.py"])
    assert blobs == {"a.py": "x = 3\n", "b.py": "y = 2\n"}
    assert git.first_parent_chain(head, 5)[-1] == head


def test_git_error_carries_command_and_survives_pickling(repo):
    import pickle

    repo.commit("c", {"x": "1\n"})
    try:
        repo.git.rev_parse("no-such-ref")
    except GitError as exc:
        again = pickle.loads(pickle.dumps(exc))
        assert again.code == exc.code and "rev-parse" in str(again)
    else:
        raise AssertionError("expected GitError")


def test_attributes_file_turns_on_the_union_driver(repo, tmp_path):
    from worktree_fleet.experiment import CHANGELOG_UNION
    from worktree_fleet.gitops import Git

    base = repo.commit("base", {"CHANGES.rst": "Unreleased\n\n- old\n"})
    a = repo.commit("a", {"CHANGES.rst": "Unreleased\n\n- entry a\n- old\n"})
    repo.checkout(base)
    b = repo.commit("b", {"CHANGES.rst": "Unreleased\n\n- entry b\n- old\n"})
    assert not repo.git.merge(a, b).clean
    attrs = tmp_path / "attrs"
    attrs.write_text(CHANGELOG_UNION)
    union = Git(repo.path, attributes_file=attrs)
    merged = union.merge(a, b)
    assert merged.clean
    text = repo.show(union.commit_tree(merged.tree, [a, b], "m"), "CHANGES.rst")
    assert "- entry a" in text and "- entry b" in text
